#!/usr/bin/env python3
"""
GENOME Alet Solver v3.2 — Main Loop

Usage:
    python solver.py problems/my_problem.yaml
    python solver.py problems/my_problem.yaml --steps 500
    python solver.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import yaml
import time
import jsonlines
import sys
import threading
from pathlib import Path
from datetime import datetime


def _load_env():
    """Загружает .env файл если ключи ещё не в окружении."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            if key and not os.environ.get(key):
                os.environ[key] = val.strip()

_load_env()

from genome_core.protocol import GenomeProtocol
from genome_core.search_tree import SearchTree
from genome_core.lantern import Lantern
from genome_core.liouville import LiouvilleChecker
from genome_core.mutations import MutationEngine
from genome_core.learner import Learner
from genome_core.thermal_guard import ThermalGuard
from engines.cas_engine import CASEngine
from engines.lean_engine import LeanEngine
from engines.llm_engine import LLMEngine
from tools.research import WebResearcher


def load_problem(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_action(action_dict):
    Path("memory").mkdir(exist_ok=True)
    with jsonlines.open("memory/session_log.jsonl", mode="a") as writer:
        action_dict["timestamp"] = datetime.now().isoformat()
        writer.write(action_dict)


def init_memory():
    Path("memory/lemmas_proved").mkdir(parents=True, exist_ok=True)
    Path("memory/hypotheses").mkdir(parents=True, exist_ok=True)
    if not Path("memory/search_tree.json").exists():
        Path("memory/search_tree.json").write_text(
            '{"step":0,"nodes":[],"lemmas":[],"killed_paradigms":[],'
            '"rep_space":null,"gap":null,"best_score":0,"problem":"","mode":"explore"}',
            encoding="utf-8",
        )
    if not Path("memory/session_log.jsonl").exists():
        Path("memory/session_log.jsonl").touch()


class StepWatchdog:
    """
    Следит за таймаутом одного шага solver.
    Если шаг занимает больше max_timeout_sec → логирует и готовится к graceful exit.
    """
    def __init__(self, max_timeout_sec: float, log_path: str):
        self.max_timeout = max_timeout_sec
        self.log_path = log_path
        self.step_start = None
        self.step_num = None
        self._lock = threading.Lock()
        self.timed_out = False

    def start_step(self, step_num: int):
        with self._lock:
            self.step_num = step_num
            self.step_start = time.time()
            self.timed_out = False

    def heartbeat(self, message: str = ""):
        """Логировать прогресс текущего шага."""
        if self.step_start is None:
            return
        elapsed = time.time() - self.step_start
        if elapsed > self.max_timeout:
            if not self.timed_out:
                self.timed_out = True
                self._log_timeout(message, elapsed)
        elif elapsed > self.max_timeout * 0.8:
            # Предупреждение за 20% до таймаута
            pct = int(100 * elapsed / self.max_timeout)
            print(f"  ⏱ Шаг {self.step_num} близок к таймауту ({pct}%)... {message[:50]}")

    def check_timeout(self) -> bool:
        """Вернуть True если шаг превысил таймаут."""
        return self.timed_out

    def _log_timeout(self, message: str, elapsed: float):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "STEP_TIMEOUT",
                    "step": self.step_num,
                    "elapsed_sec": round(elapsed, 1),
                    "max_timeout_sec": self.max_timeout,
                    "message": message[:100],
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")
        except Exception:
            pass


def deterministic_witness_edit(best_fail, artifact_dir):
    """
    Детерминированный WITNESS_GUIDED_EDIT без LLM.
    Запускает полный repair_loop (300 итераций) из tools/ramsey_repair.py.
    Каждый вызов — отдельный случайный старт от текущего best_fail графа.
    """
    import random, sys
    if "tools" not in sys.path:
        sys.path.insert(0, "tools")
    from ramsey_repair import repair_loop, load_edges, save_edges

    if not best_fail or not best_fail.get("artifact_path"):
        return {"success": False, "stage": "no_artifact", "error": "no best_fail"}

    try:
        n, edges = load_edges(best_fail["artifact_path"])
    except Exception as e:
        return {"success": False, "stage": "load", "error": str(e)}

    # Запустить repair_loop от текущего графа
    seed_edges = set(edges)
    # Возмущение только если граф НЕ является решением (partial_score < 1.0)
    if best_fail.get("partial_score", 0) < 1.0:
        perturb = random.randint(1, 5)
        all_verts = list(range(n))
        for _ in range(perturb):
            u, v = random.sample(all_verts, 2)
            e = (min(u, v), max(u, v))
            if e in seed_edges:
                seed_edges.discard(e)
            else:
                seed_edges.add(e)

    max_iters = int(os.environ.get("RAMSEY_REPAIR_ITERS", "2000"))
    ok, rn, re, vr = repair_loop(n, seed_edges, max_iters=max_iters, verbose=False)

    os.makedirs(artifact_dir, exist_ok=True)
    edges_path = os.path.join(artifact_dir, "edges.txt")
    save_edges(edges_path, rn, re)

    verdict = "OK" if ok else "FAIL"
    return {
        "success": True,
        "stage": "verified",
        "artifact_path": edges_path,
        "verdict": verdict,
        "k4_free": vr.get("k4_free"),
        "alpha_ok": vr.get("alpha_ok"),
        "k4_witness": vr.get("k4_witness"),
        "indep8_witness": vr.get("indep8_witness"),
        "k4_count": 0 if vr.get("k4_free") else 1,
        "max_indep_found": vr.get("max_indep_found", 12),
        "output": f"REPAIR_LOOP:300iters VERDICT:{verdict} n={rn} |E|={len(re)}",
        "deterministic": True,
    }


def artifact_n(path):
    """Читает n из первой строки edge-list артефакта (0 если не читается)."""
    try:
        with open(path) as f:
            return int(f.readline().strip())
    except Exception:
        return 0


def promote_ladder(solution_path, artifact_dir):
    """
    Решение при n подтверждено → построить стартовый граф для следующего n.
    Новая вершина жадно соединяется K4-safe рёбрами (давит на будущие indep-8).
    Если следующий n <= 55 и в memory/exoo_r48/edges.txt лежит проверенный граф
    Экcу (n=55, K4=0, alpha=7), прыгаем сразу к n=56: уровни 51..54 тривиальны
    (это подграфы Экcу — удаление вершин не создаёт K4 и не увеличивает alpha).
    """
    import random, itertools, sys
    if "tools" not in sys.path:
        sys.path.insert(0, "tools")
    from ramsey_repair import load_edges, save_edges, make_adj

    n, edges = load_edges(solution_path)
    edges = set(edges)

    exoo = Path("memory/exoo_r48/edges.txt")
    if n + 1 <= 55 and exoo.exists():
        n, edges = load_edges(str(exoo))
        edges = set(edges)

    new_n = n + 1
    v = new_n - 1
    adj = make_adj(new_n, edges)
    order = list(range(n))
    random.shuffle(order)
    target_deg = max(1, int(0.36 * new_n))  # ~20 при n=56, как у Экcу/circ57
    added = 0
    for u in order:
        if added >= target_deg:
            break
        common = adj[u] & adj[v]
        creates_k4 = any(b in adj[a] for a, b in itertools.combinations(common, 2))
        if not creates_k4:
            edges.add((min(u, v), max(u, v)))
            adj[u].add(v); adj[v].add(u)
            added += 1

    os.makedirs(artifact_dir, exist_ok=True)
    seed_path = os.path.join(artifact_dir, "edges.txt")
    save_edges(seed_path, new_n, edges)
    return new_n, seed_path


def ladder_promote_if_new(result, tree, step):
    """
    Если подтверждено решение на новом (ранее не решённом) n — зафиксировать
    его в memory/best_graph.txt и поднять лестницу: best_fail становится
    сидом для следующего n. Возвращает новый best_fail или None.
    """
    import shutil
    path = result.get("artifact_path")
    if not path:
        return None
    solved_n = artifact_n(path)
    if solved_n <= tree.state.get("ladder_max_solved", 0):
        return None

    shutil.copyfile(path, "memory/best_graph.txt")
    artifact_dir = f"memory/artifacts/ladder_step{step}"
    try:
        new_n, seed_path = promote_ladder(path, artifact_dir)
    except Exception as e:
        print(f"  [LADDER] promote failed: {e}")
        return None
    # Всё до new_n-1 считаем решённым (при прыжке через Экcу n=55 включительно)
    tree.state["ladder_max_solved"] = new_n - 1
    print(f"  [LADDER] n={solved_n} РЕШЕНО → новая цель n={new_n}, seed={seed_path}")

    bf = {
        "artifact_path": seed_path,
        "verdict": "FAIL",
        "k4_free": True,
        "alpha_ok": False,
        "k4_witness": None,
        "indep8_witness": None,
        "k4_count": 0,
        "max_indep_found": 12,
        "partial_score": 0.0,
        "step": step,
    }
    tree.state["best_fail"] = bf
    return bf


def graph_partial_score(result):
    """
    Непрерывный score [0..1] для частичного результата graph_edge_list.
    k4_count=0 + max_indep_found=7 -> 1.0 (решение).
    """
    k4_count = result.get("k4_count", 999)
    max_indep = result.get("max_indep_found", 12)
    k4_score = 1.0 if result.get("k4_free") else max(0.0, 1.0 - k4_count / 50.0)
    alpha_score = max(0.0, (12 - max_indep) / 5.0)  # 5 = 12-7
    alpha_score = min(1.0, alpha_score)
    return round(0.4 * k4_score + 0.6 * alpha_score, 4)


def main():
    parser = argparse.ArgumentParser(description="GENOME Alet Solver v3.2")
    parser.add_argument("problem", nargs="?", help="Path to problem YAML")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=["explore", "exploit", "verify"], default=None)
    args = parser.parse_args()

    init_memory()

    # ── Тепловой контроль ──────────────────────────────────────────────────
    thermal = ThermalGuard(
        log_path="memory/session_log.jsonl",
        print_status=True,
    )
    thermal.start()

    # ── Watchdog для таймаута шагов ────────────────────────────────────────
    solver_cfg = {}  # будет заполнено ниже
    step_watchdog = None  # инициализируем после протокола

    # Инициализация
    protocol = GenomeProtocol("config")
    solver_cfg = protocol.get_solver_config()

    # Инициализировать step_watchdog
    max_step_timeout = solver_cfg.get("max_step_timeout_seconds", 85)
    step_watchdog = StepWatchdog(max_step_timeout, "memory/session_log.jsonl")

    tree = SearchTree("memory/search_tree.json")
    cas = CASEngine(timeout=solver_cfg.get("cas_timeout_seconds", 30))
    lean = LeanEngine(timeout=solver_cfg.get("lean_timeout_seconds", 120))
    llm = LLMEngine(protocol)
    lantern = Lantern(protocol)
    liouville = LiouvilleChecker(protocol)
    mutation_engine = MutationEngine(protocol)
    learner = Learner(config_dir="config", memory_dir="memory")
    researcher = WebResearcher(llm_engine=llm)

    max_steps = args.steps or solver_cfg.get("max_steps", 10000)

    # Счётчики для детекции стены
    consecutive_kills = 0       # сколько подряд шагов ALL_KILLED
    last_research_step = -999   # шаг, на котором последний раз делали веб-поиск
    research_cooldown = solver_cfg.get("research_cooldown_steps", 30)
    wall_kill_threshold = solver_cfg.get("wall_kill_threshold", 5)
    wall_stagnation_threshold = solver_cfg.get("wall_stagnation_threshold", 25)
    web_context: dict | None = None  # последний результат веб-поиска

    # Трекинг лучшего FAIL для witness-guided редактирования
    # Загружаем из tree.state если есть (переживает --resume и свежие запуски)
    best_fail: dict | None = tree.state.get("best_fail") or None

    # Загрузка задачи
    task_type = "general"
    artifact_type = None   # "graph_edge_list" или None
    if args.problem and not args.resume:
        problem = load_problem(args.problem)
        tree.state["problem"] = problem.get("question", str(problem))
        tree.state["gap"] = problem.get("question", str(problem))
        tree.state["step"] = 0
        tree.state["nodes"] = []
        tree.state["lemmas"] = []
        tree.state["killed_paradigms"] = []
        tree.state["best_score"] = 0
        task_type = problem.get("domain", "general")
        artifact_type = problem.get("artifact_type")   # из YAML spec
        tree.state["artifact_type"] = artifact_type

        # LLM выбирает начальный REP_SPACE
        available_R = list(protocol.get_rep_spaces().keys())
        initial_R_raw = llm.ask(
            f"Задача: {tree.state['problem']}\n\n"
            f"Доступные REP_SPACE: {available_R}\n\n"
            f"Выбери начальный REP_SPACE. Верни только имя, например: R_FUNC"
        ).strip()

        # Извлечь имя пространства из ответа
        initial_R = "R_FUNC"
        for r in available_R:
            if r in initial_R_raw:
                initial_R = r
                break

        tree.state["rep_space"] = initial_R
        tree.save()
        print(f"\nProblem loaded. Initial R: {initial_R}")
        log_action({"type": "PROBLEM_LOADED", "problem": tree.state["problem"][:200],
                    "initial_R": initial_R})

    elif args.resume or tree.state.get("step", 0) > 0:
        artifact_type = tree.state.get("artifact_type")
        print(f"\nResuming from step {tree.state['step']}")

    else:
        print("Specify problem: python solver.py problems/my_problem.yaml")
        return

    if args.mode:
        tree.state["mode"] = args.mode

    print(f"Mode: {tree.state.get('mode', 'explore')}")
    print(f"Lean4 available: {lean.available}")
    print(f"LLM mode: {'API' if llm.api_key else 'human-in-loop'}")

    start_step = tree.state["step"]

    # ══════════════════════════════════════
    #         ГЛАВНЫЙ ЦИКЛ
    # ══════════════════════════════════════

    for step in range(start_step, start_step + max_steps):
        # Запустить watchdog для текущего шага
        step_watchdog.start_step(step)

        # Тепловой контроль: пауза если CPU горячий
        thermal.throttle()
        step_watchdog.heartbeat("thermal_check")

        # Проверить таймаут предыдущего шага
        if step > start_step and step_watchdog.check_timeout():
            print(f"\n  ⚠️ STEP {step-1} EXCEEDED TIMEOUT ({max_step_timeout}s) — graceful exit")
            log_action({
                "type": "TIMEOUT_EXIT",
                "step": step,
                "reason": "previous_step_timeout",
            })
            break

        t_now = thermal.current_temp()
        temp_str = f"  🌡{t_now}°C" if t_now else ""
        print(f"\n{'='*60}")
        print(f"  STEP {step}  |  R={tree.state.get('rep_space')}  |  "
              f"mode={tree.state.get('mode','explore')}  |  "
              f"lemmas={len(tree.state.get('lemmas',[]))}{temp_str}")
        print(f"{'='*60}")

        # 0. Проверить изменения протокола
        step_watchdog.heartbeat("protocol_check")
        if protocol.check_and_reload():
            print("  Protocol updated!")
            lantern = Lantern(protocol)
            liouville = LiouvilleChecker(protocol)
            mutation_engine = MutationEngine(protocol)

        # 1. LANTERN: сигналы
        step_watchdog.heartbeat("lantern_signals")
        signals = lantern.compute_signals(tree.state)
        stagnation = signals.get("stagnation", 0)
        print(f"  Signals: cx_rate={signals.get('counterexample_rate', 0):.2f}, "
              f"stagnation={stagnation}")

        # ── WALL DETECTION: упёрся в стену → веб-поиск ──
        wall_hit = (
            consecutive_kills >= wall_kill_threshold
            or stagnation >= wall_stagnation_threshold
        )
        cooldown_ok = (step - last_research_step) >= research_cooldown

        if wall_hit and cooldown_ok:
            print(f"\n  *** WALL DETECTED (kills={consecutive_kills}, "
                  f"stagnation={stagnation}) — triggering web research ***")

            # Формируем поисковый запрос из задачи + текущего пространства
            problem_text = tree.state.get("problem", "")
            current_R = tree.state.get("rep_space", "")
            search_topic = llm.ask(
                f"Задача: {problem_text[:300]}\n"
                f"Текущее пространство поиска: {current_R}\n"
                f"Мы застряли. Сформулируй поисковый запрос (на английском, 5-10 слов) "
                f"для поиска новых математических подходов в интернете. "
                f"Верни ТОЛЬКО запрос, без пояснений."
            ).strip().strip('"').strip("'")

            print(f"  [RESEARCH] Query: {search_topic!r}")

            web_context = researcher.research(
                topic=search_topic,
                problem_context=problem_text,
            )

            last_research_step = step
            consecutive_kills = 0  # сбросить счётчик после поиска

            log_action({
                "type": "WEB_RESEARCH",
                "step": step,
                "trigger": "wall",
                "kills": consecutive_kills,
                "stagnation": stagnation,
                "query": search_topic,
                "arxiv_count": len([p for p in web_context.get("arxiv_papers", []) if "error" not in p]),
                "scholar_count": len([p for p in web_context.get("scholar_papers", []) if "error" not in p]),
            })

        # 2. LIOUVILLE_TRAP
        trap = liouville.check(tree.state)
        if trap:
            print(f"  LIOUVILLE TRAP: {trap['type']}")

            old_R = tree.state["rep_space"]
            killed_spaces = [kp["space"] for kp in tree.state.get("killed_paradigms", [])]
            available_R = [
                r for r in protocol.get_rep_spaces().keys()
                if r not in killed_spaces
            ]

            if not available_R:
                print("  All REP_SPACEs exhausted! Stopping.")
                break

            new_R_raw = llm.ask(
                f"LIOUVILLE TRAP активирован!\n\n"
                f"Задача: {tree.state['problem'][:300]}\n"
                f"Текущий R: {old_R} (УБИТ — {trap['type']})\n"
                f"Уже убитые: {killed_spaces}\n"
                f"Доступные: {available_R}\n\n"
                f"Выбери новый REP_SPACE. Формат:\n"
                f"SPACE: <имя>\n"
                f"REASON: <причина>"
            )

            new_R = available_R[0]
            reason = trap["type"]
            for line in new_R_raw.split("\n"):
                if line.strip().startswith("SPACE:"):
                    candidate = line.split(":", 1)[1].strip()
                    if candidate in available_R:
                        new_R = candidate
                if line.strip().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            tree.kill_paradigm(old_R, reason, step)
            tree.state["rep_space"] = new_R
            tree.state["step"] = step + 1
            tree.save()

            log_action({"type": "SPACE_SHIFT", "from": old_R, "to": new_R,
                        "reason": reason, "step": step})

            print(f"  Killed: {old_R}")
            print(f"  Shifted to: {new_R}")
            continue

        # 3. Генерация мутаций через LLM
        n_mutations = solver_cfg.get("mutations_per_step", 15)
        mutation_prompt = mutation_engine.build_prompt(tree.state, n=n_mutations)

        # Добавить witness из лучшего FAIL (для WITNESS_GUIDED_EDIT)
        if best_fail and artifact_type == "graph_edge_list":
            k4w = best_fail.get("k4_witness")
            i8w = best_fail.get("indep8_witness")
            mutation_prompt += (
                f"\n\n=== ЛУЧШАЯ ПОПЫТКА (FAIL) ===\n"
                f"Шаг: {best_fail.get('step')}, артефакт: {best_fail.get('artifact_path')}\n"
                f"K4-free: {best_fail.get('k4_free')}, alpha<=7: {best_fail.get('alpha_ok')}\n"
            )
            if k4w:
                mutation_prompt += f"K4-witness (вершины клики): {k4w} — нужно убрать одно из рёбер\n"
            if i8w:
                mutation_prompt += f"INDEP8-witness (независимое мн-во): {i8w} — нужно добавить ребро внутри\n"
            mutation_prompt += (
                f"Предложи мутацию WITNESS_GUIDED_EDIT которая исправит это нарушение.\n"
                f"=== КОНЕЦ BEST FAIL ===\n"
            )

        # Добавить few-shot примеры из прошлых успехов
        few_shot = learner.get_few_shot_examples(n=3, task_type=task_type)
        if few_shot:
            mutation_prompt += (
                f"\n\nУспешные примеры из прошлых задач (используй как образец):\n"
                + json.dumps(few_shot, indent=2, ensure_ascii=False)
            )

        # Добавить контекст из веб-поиска (если есть)
        if web_context:
            hints = web_context.get("new_hints", [])
            approaches = web_context.get("new_approaches", [])
            brief = web_context.get("structured_brief", "")
            raw_summary = web_context.get("raw_summary", "")

            web_block = "\n\n=== НОВЫЕ ДАННЫЕ ИЗ ИНТЕРНЕТА (веб-поиск) ===\n"
            if approaches:
                web_block += "Новые подходы из литературы:\n"
                web_block += "\n".join(f"  - {a}" for a in approaches) + "\n"
            if hints:
                web_block += "Конкретные подсказки из статей:\n"
                web_block += "\n".join(f"  - {h}" for h in hints) + "\n"
            if brief and not approaches and not hints:
                # LLM не смог распарсить структурно — вставляем сырой brief
                web_block += f"Синтез из литературы:\n{brief[:800]}\n"
            elif raw_summary and not approaches and not hints:
                web_block += f"Найденные материалы:\n{raw_summary[:800]}\n"

            web_block += (
                "ЗАДАНИЕ: используй эти реальные данные из статей чтобы предложить "
                "качественно НОВЫЕ мутации, которые раньше не пробовались.\n"
                "=== КОНЕЦ ДАННЫХ ИЗ ИНТЕРНЕТА ==="
            )
            mutation_prompt += web_block

        raw_mutations = llm.ask(mutation_prompt, format="json")

        mutations = mutation_engine.parse_mutations(raw_mutations)

        if not mutations:
            preview = raw_mutations[:200].replace("\n", " ") if raw_mutations else "(empty)"
            print(f"  LLM did not return valid JSON, skipping step")
            print(f"  Raw response: {preview}")
            tree.state["step"] = step + 1
            tree.save()
            log_action({"type": "PARSE_FAIL", "step": step})
            continue

        print(f"  Generated {len(mutations)} mutations")

        # 4. TESTGEN: быстрая проверка в CAS
        scored = []
        for i, mut in enumerate(mutations):
            sympy_code = mut.get("sympy_test", "")

            # sympy_test всегда генерируется через ask_coder (Qwen / основная модель)
            if not sympy_code:
                hyp = mut.get("hypothesis", "")
                action = mut.get("concrete_action", "")
                context = hyp or action
                if context:
                    target_n = max(50, tree.state.get("ladder_max_solved", 0) + 1)
                    # Для graph_edge_list задач: строгие правила вывода
                    if artifact_type == "graph_edge_list":
                        witness_block = ""
                        if best_fail and mut.get("mutation_name") == "WITNESS_GUIDED_EDIT":
                            bf = best_fail
                            prev_edges = ""
                            if bf.get("artifact_path"):
                                try:
                                    lines = open(bf["artifact_path"]).readlines()
                                    prev_edges = "".join(lines[:1]) + "  # first line=n\n"
                                    prev_edges += f"  # {len(lines)-1} edges total\n"
                                except Exception:
                                    pass
                            k4w = bf.get("k4_witness")
                            i8w = bf.get("indep8_witness")
                            witness_block = (
                                f"\nPREVIOUS BEST ARTIFACT: {bf.get('artifact_path','?')}\n"
                                f"K4_WITNESS: {k4w} (remove one edge from this clique if not None)\n"
                                f"INDEP8_WITNESS: {i8w} (add one K4-free edge inside this set if not None)\n"
                                f"Load that file, apply the targeted fix, return new graph.\n"
                            )
                        sympy_code = llm.ask_coder(
                            f"Task: {tree.state.get('problem','')[:200]}\n"
                            f"Hypothesis: {context}\n"
                            f"{witness_block}\n"
                            f"Write Python/NetworkX code. STRICT RULES:\n"
                            f"0) Target: graph on n={target_n} vertices (current search level).\n"
                            f"1) result MUST be a dict: result = {{'n': int, 'edges': [(u,v), ...]}}\n"
                            f"2) All edges: 0 <= u < v < n. No self-loops.\n"
                            f"3) G must be K4-free: no 4 mutually adjacent vertices.\n"
                            f"4) Target: alpha(G) <= 7 (no independent set of size 8).\n"
                            f"5) No nx.Graph object in result — only the dict above.\n"
                            f"6) No markdown, no explanations, only code."
                        )
                    else:
                        sympy_code = llm.ask_coder(
                            f"Task: {tree.state.get('problem','')[:200]}\n"
                            f"Hypothesis: {context}\n\n"
                            f"Write a short Python/SymPy/NetworkX snippet to test this hypothesis. "
                            f"Rules: "
                            f"1) Assign final result to variable 'result'. "
                            f"2) For graphs always create G = nx.Graph() explicitly. "
                            f"3) Never leave G as None. "
                            f"4) Keep it fast - max {target_n} vertices, no brute force over all subsets. "
                            f"5) Only code, no markdown, no explanations."
                        )
                    sympy_code = sympy_code.strip()
                    mut["sympy_test"] = sympy_code

            # ── Детерминированный bypass для WITNESS_GUIDED_EDIT ─────────────
            if (artifact_type == "graph_edge_list"
                    and mut.get("mutation_name") == "WITNESS_GUIDED_EDIT"
                    and best_fail):
                mut_name = "WITNESS_GUIDED_EDIT[DET]"
                print(f"  [{i+1}/{len(mutations)}] {mut_name}...", end=" ")
                artifact_dir = f"memory/artifacts/step{step}_mut{i}"
                result = deterministic_witness_edit(best_fail, artifact_dir)
                verdict = result.get("verdict")
                pscore = graph_partial_score(result)
                if result.get("success") and verdict == "OK":
                    score = lantern.score_mutation(mut, {"success": True, "output": result["output"]}, signals)
                    scored.append((mut, score, result))
                    print(f"VERIFIED_OK score={score:.2f}")
                    new_bf = ladder_promote_if_new(result, tree, step)
                    if new_bf:
                        best_fail = new_bf
                else:
                    reason = result.get("error") or f"VERDICT:{verdict}"
                    print(f"FAIL partial_score={pscore:.3f} k4={result.get('k4_free')} a={result.get('alpha_ok')}")
                    if (result.get("artifact_path") and result.get("stage") == "verified"
                            and artifact_n(result["artifact_path"]) > tree.state.get("ladder_max_solved", 0)):
                        new_ps = pscore
                        old_ps = best_fail.get("partial_score", 0)
                        if new_ps >= old_ps:
                            best_fail = {
                                "artifact_path": result["artifact_path"],
                                "verdict": verdict,
                                "k4_witness": result.get("k4_witness"),
                                "indep8_witness": result.get("indep8_witness"),
                                "k4_free": result.get("k4_free"),
                                "alpha_ok": result.get("alpha_ok"),
                                "k4_count": result.get("k4_count", 0),
                                "max_indep_found": result.get("max_indep_found", 12),
                                "partial_score": new_ps,
                                "step": step,
                            }
                            tree.state["best_fail"] = best_fail
                            print(f"  [best_fail updated partial_score={new_ps:.3f}]")
                    killed_node = {
                        "step": step,
                        "mutation_name": "WITNESS_GUIDED_EDIT",
                        "rep_space": tree.state["rep_space"],
                        "hypothesis": "deterministic witness edit",
                        "killed": True,
                        "kill_reason": str(reason)[:120],
                        "artifact_path": result.get("artifact_path"),
                        "verdict": verdict,
                        "k4_witness": result.get("k4_witness"),
                        "indep8_witness": result.get("indep8_witness"),
                        "partial_score": pscore,
                        "score": 0,
                    }
                    tree.add_node(killed_node)
                    learner.record_step(killed_node, task_type=task_type)
                continue   # skip LLM/CAS for this mutation
            # ─────────────────────────────────────────────────────────────────

            if not sympy_code:
                continue

            mut_name = mut.get("mutation_name", "?")[:30]
            print(f"  [{i+1}/{len(mutations)}] Testing: {mut_name}...", end=" ")

            artifact_dir = f"memory/artifacts/step{step}_mut{i}"
            try:
                if artifact_type == "graph_edge_list":
                    result = cas.run_graph_task(sympy_code, artifact_dir)
                else:
                    result = cas.run_sympy(sympy_code)
            except Exception as e:
                result = {"success": False, "error": f"{type(e).__name__}: {e}"}

            verdict = result.get("verdict")   # "OK" / "FAIL" / None

            if artifact_type == "graph_edge_list":
                # fit=1 ТОЛЬКО при VERDICT OK
                if result.get("success") and verdict == "OK":
                    score = lantern.score_mutation(mut, {"success": True, "output": result["output"]}, signals)
                    scored.append((mut, score, result))
                    print(f"VERIFIED_OK score={score:.2f} artifact={result.get('artifact_path','?')}")
                    new_bf = ladder_promote_if_new(result, tree, step)
                    if new_bf:
                        best_fail = new_bf
                else:
                    reason = result.get("error") or f"VERDICT:{verdict} k4={result.get('k4_free')} a={result.get('alpha_ok')}"
                    print(f"FAIL {str(reason)[:60]}")
                    # Обновить best_fail если есть артефакт (для witness-guided edit)
                    if (result.get("artifact_path") and result.get("stage") == "verified"
                            and artifact_n(result["artifact_path"]) > tree.state.get("ladder_max_solved", 0)):
                        pscore = graph_partial_score(result)
                        old_ps = best_fail.get("partial_score", 0) if best_fail else 0
                        if pscore >= old_ps:
                            best_fail = {
                                "artifact_path": result["artifact_path"],
                                "verdict": verdict,
                                "k4_witness": result.get("k4_witness"),
                                "indep8_witness": result.get("indep8_witness"),
                                "k4_free": result.get("k4_free"),
                                "alpha_ok": result.get("alpha_ok"),
                                "k4_count": result.get("k4_count", 0),
                                "max_indep_found": result.get("max_indep_found", 12),
                                "partial_score": pscore,
                                "step": step,
                            }
                            tree.state["best_fail"] = best_fail
                            print(f"  [best_fail updated: k4_free={result.get('k4_free')} "
                                  f"alpha_ok={result.get('alpha_ok')} partial_score={pscore:.3f}]")
                    killed_node = {
                        "step": step,
                        "mutation_name": mut.get("mutation_name", "?"),
                        "rep_space": tree.state["rep_space"],
                        "hypothesis": mut.get("hypothesis", ""),
                        "killed": True,
                        "kill_reason": str(reason)[:120],
                        "artifact_path": result.get("artifact_path"),
                        "verdict": verdict,
                        "k4_witness": result.get("k4_witness"),
                        "indep8_witness": result.get("indep8_witness"),
                        "score": 0,
                    }
                    tree.add_node(killed_node)
                    learner.record_step(killed_node, task_type=task_type)
            elif result["success"]:
                score = lantern.score_mutation(mut, result, signals)
                scored.append((mut, score, result))
                print(f"OK score={score:.2f}")
            else:
                err = result.get("error", "")[:50]
                print(f"FAIL {err}")
                killed_node = {
                    "step": step,
                    "mutation_name": mut.get("mutation_name", "?"),
                    "rep_space": tree.state["rep_space"],
                    "hypothesis": mut.get("hypothesis", ""),
                    "killed": True,
                    "kill_reason": result.get("error", "CAS failed"),
                    "score": 0,
                }
                tree.add_node(killed_node)
                learner.record_step(killed_node, task_type=task_type)

        if not scored:
            print("  All mutations killed by CAS")
            consecutive_kills += 1
            tree.state["step"] = step + 1
            tree.save()
            log_action({"type": "ALL_KILLED", "step": step, "consecutive": consecutive_kills})
            continue

        # 5. Выбор лучшей мутации
        scored.sort(key=lambda x: x[1], reverse=True)
        best_mut, best_score, best_result = scored[0]

        # Граф-задача решена → сохранить и выйти
        if artifact_type == "graph_edge_list" and best_result.get("verdict") == "OK":
            sol_path = "memory/solution_graph.txt"
            import shutil
            shutil.copy(best_result["artifact_path"], sol_path)
            print(f"\n  *** GRAPH SOLUTION FOUND! ***")
            print(f"  artifact: {best_result['artifact_path']}")
            print(f"  saved to: {sol_path}")
            tree.state["step"] = step + 1
            tree.state["best_score"] = 1.0
            tree.state["solved_artifact"] = sol_path
            tree.add_node({"step": step, "mutation_name": best_mut.get("mutation_name", "?"),
                           "rep_space": tree.state["rep_space"],
                           "score": best_score, "killed": False,
                           "verdict": "OK", "artifact_path": best_result["artifact_path"]})
            tree.save()
            learner.after_task(solved=True, step=step)
            return

        consecutive_kills = 0  # успешная мутация — сбросить счётчик стены
        print(f"\n  BEST: {best_mut.get('mutation_name', '?')} (score={best_score:.3f})")
        print(f"  Action: {best_mut.get('concrete_action', '')[:80]}")

        # 6. Lean верификация (если доступен и score достаточный)
        lean_result = {"proved": False, "errors": ["skipped"], "warnings": []}

        if lean.available and best_score > 0.5:
            lean_prompt = (
                f"Сформулируй на Lean4 (с import Mathlib):\n\n"
                f"{best_mut.get('hypothesis', '')}\n\n"
                f"CAS подтверждение: {best_result.get('output', '')}\n\n"
                f"Верни только .lean код."
            )
            lean_code = llm.ask(lean_prompt)
            lean_result = lean.check_proof(lean_code)

            if lean_result["proved"]:
                print("  PROVED IN LEAN!")
                lean_path = f"memory/lemmas_proved/lemma_step{step}.lean"
                Path(lean_path).write_text(lean_code, encoding="utf-8")
                tree.add_lemma({
                    "step": step,
                    "statement": best_mut.get("hypothesis", ""),
                    "mutation": best_mut.get("mutation_name", ""),
                    "lean_file": lean_path,
                    "score": best_score,
                })
            else:
                print(f"  Lean errors: {lean_result['errors'][:3]}")

        # 7. Обновить дерево
        node_dict = {
            "step": step,
            "mutation_name": best_mut.get("mutation_name", "?"),
            "concrete_action": best_mut.get("concrete_action", ""),
            "rep_space": tree.state["rep_space"],
            "hypothesis": best_mut.get("hypothesis", ""),
            "cas_output": best_result.get("output", ""),
            "lean_proved": lean_result.get("proved", False),
            "score": best_score,
            "killed": False,
            # artifact fields (None для non-graph задач)
            "artifact_path": best_result.get("artifact_path"),
            "verdict": best_result.get("verdict"),
            "witness": {
                "k4": best_result.get("k4_witness"),
                "indep8": best_result.get("indep8_witness"),
            } if best_result.get("verdict") else None,
        }
        tree.add_node(node_dict)

        # Learner: записать шаг для самообучения
        learner.record_step(node_dict, task_type=task_type)

        if best_score > tree.state.get("best_score", 0):
            tree.state["best_score"] = best_score

        tree.state["step"] = step + 1
        tree.save()

        log_action({
            "type": "STEP_COMPLETE",
            "step": step,
            "mutation": best_mut.get("mutation_name", "?"),
            "score": best_score,
            "proved": lean_result.get("proved", False),
        })

        # 8. Мета-рефлексия каждые 50 шагов
        if step > 0 and step % 50 == 0:
            print(f"\n  META-REFLECTION at step {step}")
            summary = tree.summary()
            meta_prompt = (
                f"Мета-рефлексия. Проанализируй прогресс:\n\n"
                f"{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
                f"Последние 50 узлов:\n"
                f"{json.dumps(tree.state['nodes'][-50:], indent=2, ensure_ascii=False)[:3000]}\n\n"
                f"Ответь:\n"
                f"1) Есть ли прогресс?\n"
                f"2) Нужно ли сменить REP_SPACE?\n"
                f"3) Какие мутации работают, какие нет?\n"
                f"4) Рекомендация: продолжать / сменить стратегию?\n"
                f"5) Нужно ли изменить протокол?"
            )
            meta = llm.ask(meta_prompt)
            print(f"  {meta[:300]}...")
            log_action({"type": "META_REFLECTION", "step": step,
                        "summary": summary, "reflection": meta[:500]})

    # Финал
    thermal.stop()

    print(f"\n{'='*60}")
    print(f"  SOLVER FINISHED")
    print(f"  Steps: {tree.state['step']}")
    print(f"  Lemmas proved: {len(tree.state.get('lemmas', []))}")
    print(f"  Best score: {tree.state.get('best_score', 0):.3f}")
    print(f"{'='*60}")

    # Learner: обновить конфиги на основе накопленной статистики
    solved = len(tree.state.get("lemmas", [])) > 0
    learner.after_task(solved=solved, step=tree.state["step"])


if __name__ == "__main__":
    main()

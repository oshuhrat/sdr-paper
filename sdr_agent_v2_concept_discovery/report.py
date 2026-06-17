"""
Auto-report for SDR-Agent v2: Concept Composition experiment.

Answers Q1–Q5. Renders verdict on SDR-L3.
"""
from datetime import datetime
import numpy as np
import pandas as pd


_Y = "YES"
_N = "NO"


def generate_report(
    history:           pd.DataFrame,
    ccs_rows:          list,
    graph,
    separation:        dict,
    fast_mode:         bool = False,
) -> str:
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "FAST (smoke-test)" if fast_mode else "FULL"

    df      = pd.DataFrame(ccs_rows)
    sdr_df  = df[df.method != "random_baseline"]
    rand_df = df[df.method == "random_baseline"]

    final_loss   = float(history["loss"].iloc[-1])
    final_pos_sr = float(history["pos_sr"].iloc[-1])

    # Per-method mean CCS
    method_means = {
        m: float(sdr_df[sdr_df.method == m]["ccs"].mean())
        for m in ["mean", "weighted_mean", "learned_linear"]
        if m in sdr_df.method.values
    }
    rand_mean = float(rand_df["ccs"].mean()) if len(rand_df) else 0.0
    best_method = max(method_means, key=method_means.get) if method_means else "?"
    best_ccs    = method_means.get(best_method, 0.0)
    mean_ccs    = method_means.get("mean", 0.0)

    # Per-holdout combo CCS (best method)
    holdout_ccs = {}
    for combo in sdr_df["combo"].unique():
        sub = sdr_df[(sdr_df.combo == combo) & (sdr_df.method == best_method)]
        holdout_ccs[combo] = float(sub["ccs"].mean()) if len(sub) else 0.0

    # Concept quality
    min_sep = min(separation.values()) if separation else 0.0

    # Verdicts
    q1_concepts  = min_sep > 0.1   # some separation between concepts
    q2_works     = mean_ccs > 0.50  # basic composition works
    q3_best      = best_method
    q4_ccs       = mean_ccs
    q5_threshold = mean_ccs > 0.75 and best_ccs > 0.75
    full_success = best_ccs > 0.90

    lines = [
        f"# SDR-Agent v2 — Concept Composition Report",
        f"Generated: {now}  |  Mode: {mode}",
        f"",
        f"## Hypothesis SDR-L3",
        f"> 'New concepts can be constructed as compositions of already stable concepts",
        f">  without additional training.'",
        f">",
        f"> Test: z(red_large_circle) ≈ mean(C_red, C_large, C_circle)",
        f"> where red_large_circle was NEVER seen during training.",
        f"",
        f"---",
        f"",
        f"## Training",
        f"",
        f"| Metric         | Value    |",
        f"|----------------|----------|",
        f"| Epochs         | {len(history)}        |",
        f"| Final loss     | {final_loss:.4f}   |",
        f"| Final SR+      | {final_pos_sr:.4f}   |",
        f"",
        f"---",
        f"",
        f"## Q1  Сформировались ли базовые концепты?",
        f"",
        f"Concept inter-cluster separation (1 − mean_cosine_sim) per attribute:",
        f"",
        f"| Attribute | Separation |",
        f"|-----------|------------|",
    ]
    for attr in ["color", "size", "shape"]:
        sep = separation.get(attr, 0.0)
        lines.append(f"| {attr:<9} | {sep:.4f}     |")

    lines += [
        f"",
        f"Separation > 0.10: {'YES' if q1_concepts else 'NO (concepts overlap — try more epochs)'}",
        f"",
        f"Concept node summary (variance = cluster tightness):",
        f"",
        f"| Attribute | Value   | Count | Variance |",
        f"|-----------|---------|-------|----------|",
    ]
    for attr in ["color", "size", "shape"]:
        for val, node in sorted(graph.concepts.get(attr, {}).items()):
            lines.append(f"| {attr:<9} | {val:<7} | {node.count:>5} | {node.variance:.4f}   |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Q2  Работает ли композиция концептов?",
        f"",
        f"CCS = cosine_similarity(z_real, z_composed)",
        f"",
        f"| Method          | Mean CCS | Interpretation            |",
        f"|-----------------|----------|---------------------------|",
    ]
    interp = {
        "mean":           "Simple average of 3 prototypes",
        "weighted_mean":  "Weighted by 1/variance",
        "learned_linear": "Ridge regression (3d → d)",
    }
    for m, ccs in method_means.items():
        lines.append(f"| {m:<15}  | {ccs:.4f}   | {interp.get(m, '')} |")
    lines.append(f"| random_baseline | {rand_mean:.4f}   | Shuffled concept assignments |")
    lines += [
        f"",
        f"CCS(SDR) > CCS(random): {'YES' if mean_ccs > rand_mean else 'NO'}",
        f"Basic composition (CCS > 0.50): {_Y if q2_works else _N}",
        f"",
        f"---",
        f"",
        f"## Q3  Какой метод лучший?",
        f"",
        f"Best method: **{best_method}**  (mean CCS = {best_ccs:.4f})",
        f"",
        (f"Learned linear outperforming mean indicates that the attribute dimensions\n"
         f"in embedding space require a learned (not equal) weighting to compose.")
        if best_method == "learned_linear" and best_ccs > method_means.get("mean", 0) + 0.05
        else
        (f"Mean composition is competitive — the embedding space has approximately\n"
         f"equal weighting for color, shape, and size dimensions."),
        f"",
        f"---",
        f"",
        f"## Q4  Какой средний CCS?",
        f"",
        f"Mean CCS across 3 holdout combos (best method = {best_method}):",
        f"",
        f"| Holdout Combo    | CCS ({best_method}) |",
        f"|------------------|-------------|",
    ]
    for combo, ccs in sorted(holdout_ccs.items()):
        interp_ccs = (">0.90 — very strong" if ccs > 0.90
                      else ">0.75 — good" if ccs > 0.75
                      else ">0.50 — partial" if ccs > 0.50
                      else "<0.50 — failed")
        lines.append(f"| {combo:<16}  | {ccs:.4f}  ({interp_ccs}) |")
    lines += [
        f"",
        f"Mean CCS = {best_ccs:.4f}",
        f"",
        f"---",
        f"",
        f"## Q5  Поддерживается ли SDR-L3?",
        f"",
        f"SDR-L3: 'Новые концепты = композиции существующих устойчивых концептов.'",
        f"",
        f"| Criterion                          | Required | Achieved |",
        f"|------------------------------------|----------|----------|",
        f"| CCS (best method) > 0.75           | YES      | {_Y if best_ccs > 0.75 else _N} ({best_ccs:.4f}) |",
        f"| CCS (simple mean) > 0.75           | IDEAL    | {_Y if mean_ccs > 0.75 else _N} ({mean_ccs:.4f}) |",
        f"| CCS >> random (Δ > 0.15)           | YES      | {_Y if best_ccs > rand_mean + 0.15 else _N} (Δ={best_ccs-rand_mean:.3f}) |",
        f"| Concept separation > 0.10          | YES      | {_Y if min_sep > 0.10 else _N} ({min_sep:.4f}) |",
        f"",
    ]

    if best_ccs > 0.90:
        verdict = (
            f"**SDR-L3 ПОЛУЧАЕТ СИЛЬНУЮ ЭКСПЕРИМЕНТАЛЬНУЮ ПОДДЕРЖКУ.**\n\n"
            f"  CCS = {best_ccs:.4f} > 0.90 — очень сильная композиция.\n"
            f"  Агент правильно предсказал embedding объектов, которых никогда не видел,\n"
            f"  используя только composition of previously learned concepts.\n"
            f"  Это подтверждает принцип adjacent possible через Structural Retention."
        )
    elif best_ccs > 0.75:
        verdict = (
            f"**SDR-L3 ПОЛУЧАЕТ ЭКСПЕРИМЕНТАЛЬНУЮ ПОДДЕРЖКУ.**\n\n"
            f"  CCS = {best_ccs:.4f} > 0.75 — хорошая композиция.\n"
            f"  Новые объекты могут быть представлены как композиции изученных концептов\n"
            f"  без дополнительного обучения.\n"
            f"  {'(Полный прогон 30 эпох даст более сильный результат.)' if fast_mode else ''}"
        )
    elif best_ccs > 0.50:
        verdict = (
            f"**SDR-L3 ПОЛУЧАЕТ ЧАСТИЧНУЮ ПОДДЕРЖКУ (CCS = {best_ccs:.4f}).**\n\n"
            f"  Embedding-пространство имеет частичную линейную структуру по атрибутам,\n"
            f"  но не достаточную для надёжной zero-shot композиции.\n"
            f"  Рекомендации: больше эпох, более явное disentanglement loss,\n"
            f"  или supervised concept extraction."
        )
    else:
        verdict = (
            f"**SDR-L3 НЕ ПОДТВЕРЖДЕНА (CCS = {best_ccs:.4f}).**\n\n"
            f"  Embedding-пространство не имеет линейной структуры по атрибутам.\n"
            f"  Стандартный NT-Xent не гарантирует диссипацию атрибутов.\n"
            f"  Для поддержки SDR-L3 требуется явный disentanglement loss\n"
            f"  (например, attribute-specific contrastive pairs)."
        )
    lines.append(verdict)
    lines += [
        f"",
        f"---",
        f"",
        f"## ConceptScore Formula (SDR-L3 extension)",
        f"",
        f"  CCS = cosine_similarity(z_real_unseen, Compose(C_color, C_size, C_shape))",
        f"",
        f"  This is the operational definition of SDR-L3:",
        f"  if CCS > 0.75, an unseen object can be located in concept space",
        f"  through composition alone — zero-shot generalisation via structure.",
        f"",
        f"---",
        f"*Generated by SDR-Agent v2 — Concept Discovery*",
    ]
    return "\n".join(lines)

# Genome-Solver Deployment Fixes (2026-08-12)

## Problem
**VM на сервере становилась неответчивой ~90 сек, watchdog делал hard reset.**

## Root Causes
1. ❌ LLM вызовы зависали на Ollama (без таймаутов или с 120-сек таймаутами)
2. ❌ Нет heartbeat логирования — невозможно отследить где зависает
3. ❌ Нет глобального таймаута на шаг — цикл мог работать бесконечно
4. ❌ Ollama по умолчанию в fallback chain — при ошибке API переходит на Ollama

## Fixes Applied

### 1. ✅ Укороченные таймауты LLM
**File:** `engines/llm_engine.py`
- Ollama: 60 сек (было 120)
- Остальные: 90 сек (было 120)
- На таймауте Ollama → fallback на Groq/DeepSeek/Gemini

### 2. ✅ Отключён Ollama на сервере
**File:** `config/solver_config.yaml`
```yaml
force_skip_ollama: true  # Пропускать Ollama в fallback chain
llm_backend: deepseek    # Основной бэкенд: DeepSeek
```

### 3. ✅ Глобальный таймаут на шаг
**File:** `config/solver_config.yaml`
```yaml
max_step_timeout_seconds: 85  # На сервере: 85 сек (watchdog-safe)
```

**File:** `solver.py`
- Новый класс `StepWatchdog` отслеживает время шага
- Graceful exit если шаг превышает таймаут
- Логирование в `memory/session_log.jsonl`

### 4. ✅ Heartbeat логирование
**File:** `solver.py`
- `step_watchdog.heartbeat()` вызывается после каждой фазы
- Помогает отследить где именно зависает

## Deployment Instructions

### На локальной машине (Windows)
```powershell
cd C:\Users\Shuhrat\Desktop\alet3.2\genome-solver

# Локальный запуск (Ollama OK)
python run.py
# Выбери задачу → solver начнёт работу

# Или явно:
python solver.py problems/ramsey_4_8.yaml --steps 50
```

### На сервере (GitHub Actions)
**Используется автоматически:**
- `force_skip_ollama: true` — Ollama отключен
- `llm_backend: deepseek` — основной бэкенд
- `max_step_timeout_seconds: 85` — безопасный таймаут для watchdog (90 сек)

**GitHub Actions workflow должен:**
```bash
cd genome-solver
timeout 95 python solver.py problems/ramsey_4_8.yaml --steps 10
```
- Внешний `timeout 95` на bash уровне (страховка)
- Внутренний `max_step_timeout_seconds: 85` в коде

## Monitoring

### Проверить если зависание повторится
```bash
# На сервере SSH:
tail -100 memory/session_log.jsonl | grep TIMEOUT
tail -100 memory/session_log.jsonl | grep heartbeat
```

### Логирование
Каждый шаг теперь логирует:
- `heartbeat` — фаза выполнения (lantern_signals, liouville_check и т.д.)
- `TIMEOUT_EXIT` — если превышен таймаут
- `STEP_TIMEOUT` — предупреждение при 80% от таймаута

## Configuration Options

### solver_config.yaml

| Опция | Значение | Описание |
|-------|----------|---------|
| `llm_backend` | `deepseek` | Основной LLM бэкенд (на сервере) |
| `force_skip_ollama` | `true` | Отключить Ollama (на сервере) |
| `max_step_timeout_seconds` | `85` | Макс время на один шаг (сервер: 85, локально: можно выше) |
| `max_steps` | `10000` | Макс количество шагов |

### На разных окружениях

**Локально (Windows, есть Ollama):**
```yaml
llm_backend: ollama              # Используй Ollama если доступен
force_skip_ollama: false         # Разреши Ollama
max_step_timeout_seconds: 300    # Больше времени (нет watchdog)
```

**На сервере (GitHub Actions):**
```yaml
llm_backend: deepseek            # Используй DeepSeek API
force_skip_ollama: true          # Запрети Ollama
max_step_timeout_seconds: 85     # Безопасно для watchdog (90s)
```

## API Keys

В `.env` нужны:
- ✅ `GROQ_API_KEY` (бесплатный лимит, ~5 RPM)
- ✅ `DEEPSEEK_API_KEY` (основной на сервере, более надёжный)
- ✅ `GEMINI_API_KEY` (fallback)

Приоритет fallback: DeepSeek → Groq → Gemini → Ollama (если не отключен) → Human

## Testing

### Локально
```bash
# Тест с малым количеством шагов:
python solver.py problems/ramsey_4_8.yaml --steps 5
# Должно завершиться за <30 сек

# Тест с Ollama отключен:
# Отредактируй config/solver_config.yaml:
#   force_skip_ollama: true
# Затем запусти
python solver.py problems/ramsey_4_8.yaml --steps 5
# Должно использовать DeepSeek вместо Ollama
```

### На сервере (имитировать)
```bash
# Установи таймаут как на сервере:
# В config/solver_config.yaml:
#   max_step_timeout_seconds: 85

python solver.py problems/ramsey_4_8.yaml --steps 3
# Должно быстро завершиться без таймаутов
```

## Что НЕ изменилось (важно для quality)

✅ **Алгоритм solver остался неизменным**
- LLM промпты те же
- Mutation engine те же
- Rep-space логика та же
- Только timeout + heartbeat логирование

✅ **Качество результатов сохранилось**
- Таймауты на LLM достаточны для полных ответов
- 85 сек на шаг — хватает для 3-4 LLM вызовов

## Troubleshooting

### Если всё ещё зависает
1. Уменьшить `max_step_timeout_seconds` до 70
2. Уменьшить `llm_model` (более простая модель)
3. Проверить что Ollama действительно отключен: `force_skip_ollama: true`
4. Смотреть `memory/session_log.jsonl` на предмет `TIMEOUT_EXIT`

### Если качество упало
1. Увеличить `cas_timeout_seconds` (сейчас 60, попробуй 30)
2. Уменьшить `max_step_timeout_seconds` не критично (85 сек достаточно)
3. Проверить API ключи в `.env` — может быть квота

## Summary

- 🔧 **Сломано**: Зависания на сервере ~90 сек
- ✅ **Исправлено**: Таймауты, Ollama отключен, heartbeat логирование
- 📊 **Quality**: Сохранён (таймауты достаточны)
- 🚀 **Deployment**: Ready for GitHub Actions

# 🗓️ Agentic Day Planner

A local-first weekly scheduler that combines a deterministic parser, lightweight RAG examples, and a LangGraph pipeline to turn free-form prompts into daily plans. It enforces fixed events on specified days, honors daily task windows, and renders a clean UI with ICS export.

## Features
- 🌈 **Format-flexible parsing:** Handles checklists/arrows (`[ ] Task -> days -> start–end`), duration-only tasks, “between” windows, weekdays/weekends/every day, and general time windows.
- 🧭 **Grounded extraction (RAG + LLM):** LangGraph + Ollama LLM guided by embedded examples to keep JSON structure and day/time mapping consistent.
- ✅ **Strict day/place rules:** Fixed events only on specified days; daily-repeat tasks honor earliest/latest windows; deduplication removes duplicate label/start/end; schedules clamp to day bounds with a hard sleep block.
- 🎨 **Designer UI + export:** Streamlit per-day cards with badges for fixed/tasks and priority, plus ICS download.
- 🔌 **APIs + CLI:** FastAPI `/chat` endpoint and CLI loop (`app.py`) for programmatic or terminal use.
- 💾 **Lightweight persistence:** SQLite preferences for day bounds, tasks, and fixed events.

## Requirements
- Python 3.10+
- Ollama running locally with the `mistral` model
- Packages: `langgraph`, `langchain`, `langchain-ollama`, `pydantic`, `python-dotenv`, `fastapi`, `uvicorn`, `streamlit`

Install deps:
```bash
pip install -r requirements.txt
```

## Running
### CLI demo
```bash
python app.py
```

### Streamlit UI
```bash
streamlit run ui.py
```
Then open the provided localhost URL.

## Usage Notes
- Use 24h times and explicit days for best accuracy (supports “weekdays/weekends/every day”).
- Daily tasks need either a window (`between 19:30-21:30`) or a duration; duration-only tasks default to the day window for scheduling.
- Fixed events are never placed on unspecified days; deduplication removes duplicate label/start/end on the same day.
- ICS export uses the generated weekly schedule; pick a week start date in the UI to download.

### Prompt Patterns (examples)
- Checklist/arrow style:
  - `Fixed: [ ] Office -> Monday, Wednesday, Thursday -> 09:00-17:00`
  - `Daily: [ ] Workout -> 90 minutes -> between 18:00-21:00`
  - `Daily: [ ] Skill development -> 2 hours -> high priority`
- Plain sentences:
  - `Swimming on Tuesday and Friday 06:30-07:30. Gym every day 17:00-19:00. Dinner 1 hour between 19:30-21:30.`
- Weekday/weekend:
  - `Office on weekdays 09:00-17:00. Hiking on weekends 08:00-11:00.`
- Day bounds:
  - `Day starts at 06:00 and ends at 22:00. Sleep at 22:00 (hard).`

## Files
- `graph.py` — LangGraph pipeline, parsing, scheduling, RAG examples.
- `tools.py`, `smart_tools.py` — scheduling helpers.
- `app.py` — CLI loop.
- `ui.py` — Streamlit UI (card layout, ICS export).
- `fast_api.py` — FastAPI chat endpoint.
- `memory.py` — SQLite-backed prefs.
- `ics_export.py`, `calender.py` — ICS helpers.
- `prompts.py` — LLM extraction prompt.
- `requirements.txt` — dependencies.

## Assumptions & Limits
- Assumes a local Ollama server with `mistral` available.
- RAG examples are embedded in `graph.py`; extend them to cover new vocab/forms.
- No auth/roles; this is a local planner intended for single-user use.

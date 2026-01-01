# Agentic Day Planner

A local-first weekly scheduler that combines a deterministic parser, lightweight RAG examples, and a LangGraph pipeline to turn free-form prompts into daily plans. It enforces fixed events on specified days, honors daily task windows, and renders a clean UI with ICS export.

## Features
- Deterministic prompt parsing for checklists/arrows (e.g., `[ ] Task -> days -> start–end`, duration-only, and windowed tasks).
- LangGraph + Ollama LLM extraction with embedded examples for structure.
- Scheduling pipeline: fixed events per day, daily repeat tasks with time windows, deduplication, and clamping to day bounds with a hard sleep block.
- Streamlit “designer” UI (card-based per day) plus ICS download.
- Lightweight SQLite preference storage.

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

### FastAPI endpoint
```bash
uvicorn fast_api:app --reload --port 8000
```

## Usage Notes
- Use 24h times and explicit days for best accuracy (supports “weekdays/weekends/every day”).
- Daily tasks need either a window (`between 19:30-21:30`) or a duration; duration-only tasks default to the day window for scheduling.
- Fixed events are never placed on unspecified days; deduplication removes duplicate label/start/end on the same day.
- ICS export uses the generated weekly schedule; pick a week start date in the UI to download.

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

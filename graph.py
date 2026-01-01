# graph.py
from __future__ import annotations

import json
import re
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

# ✅ Use the new package (avoid deprecated ChatOllama warnings)
from langchain_ollama import ChatOllama

from prompts import EXTRACT_PROMPT
from memory import load_prefs, save_prefs
from tools import build_draft_schedule, check_conflicts, auto_fix_conflicts, to_min, to_hhmm
from smart_tools import inject_breaks, promote_deep_work, clamp_to_day_end

from pydantic import BaseModel, Field
from typing import Literal


# -----------------------------
# Constants
# -----------------------------
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

DAY_ALIASES = {
    "monday": "mon", "mon": "mon",
    "tuesday": "tue", "tue": "tue",
    "wednesday": "wed", "wed": "wed",
    "thursday": "thu", "thu": "thu",
    "friday": "fri", "fri": "fri",
    "saturday": "sat", "sat": "sat",
    "sunday": "sun", "sun": "sun",
}


# -----------------------------
# Pydantic extraction schema
# -----------------------------
class FixedEvent(BaseModel):
    label: str
    start: str
    end: str


class Task(BaseModel):
    title: str
    duration_min: int = Field(ge=10, le=480)
    priority: Literal["low", "medium", "high"] = "medium"


class ExtractedWeekly(BaseModel):
    day_start: str = "06:00"
    day_end: str = "22:00"
    commute_before_min: int = 0
    commute_after_min: int = 0

    weekly_fixed: Dict[str, List[FixedEvent]] = Field(default_factory=lambda: {d: [] for d in DAYS})
    tasks: List[Task] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


# -----------------------------
# LangGraph State
# -----------------------------
class SchedulerState(TypedDict, total=False):
    user_id: str
    user_request: str

    # extracted + normalized
    day_start: str
    day_end: str
    commute_before_min: int
    commute_after_min: int
    weekly_fixed: Dict[str, List[Dict[str, Any]]]
    tasks: List[Dict[str, Any]]

    weekly_schedule: Dict[str, List[Dict[str, Any]]]
    notes: List[str]

    extracted: Dict[str, Any]


# -----------------------------
# Global LLM (faster: keeps warm)
# -----------------------------
OLLAMA_LLM = ChatOllama(
    model="mistral",
    temperature=0.2,
    format="json",
    keep_alive="10m",
)


# -----------------------------
# Lightweight RAG examples
# -----------------------------
def load_rag_examples() -> list[dict]:
    """Embed example set directly; can be extended without external file."""
    return [
        {
            "desc": "Checklist style daily tasks with evening windows",
            "prompt": "Goal: weekly schedule. Day starts 06:00, ends 22:00. Fixed: Office Tue/Thu 09:00-17:00. Daily: Coding 2 hours high; Workout 90 minutes evening only (18:00-21:00); Dinner 45 minutes between 19:00-21:30.",
            "json": {
                "day_start": "06:00",
                "day_end": "22:00",
                "weekly_fixed": {
                    "tue": [{"label": "Office", "start": "09:00", "end": "17:00"}],
                    "thu": [{"label": "Office", "start": "09:00", "end": "17:00"}]
                },
                "tasks": [
                    {"title": "Skill development", "duration_min": 120, "priority": "high", "daily_repeat": True},
                    {"title": "Workout", "duration_min": 90, "priority": "medium", "earliest_start": "18:00", "latest_end": "21:00", "daily_repeat": True},
                    {"title": "Dinner", "duration_min": 45, "priority": "low", "earliest_start": "19:30", "latest_end": "21:30", "daily_repeat": True}
                ]
            }
        },
        {
            "desc": "Office weekdays, swim Mon/Wed, yoga daily evenings",
            "prompt": "Swim on Mon and Wed 07:00-08:00. Office on weekdays 09:00-17:00. Yoga every day 18:00-19:00.",
            "json": {
                "day_start": "06:00",
                "day_end": "22:00",
                "weekly_fixed": {
                    "mon": [{"label": "Swim", "start": "07:00", "end": "08:00"}, {"label": "Office", "start": "09:00", "end": "17:00"}],
                    "wed": [{"label": "Swim", "start": "07:00", "end": "08:00"}, {"label": "Office", "start": "09:00", "end": "17:00"}],
                    "tue": [{"label": "Office", "start": "09:00", "end": "17:00"}],
                    "thu": [{"label": "Office", "start": "09:00", "end": "17:00"}],
                    "fri": [{"label": "Office", "start": "09:00", "end": "17:00"}]
                },
                "tasks": [
                    {"title": "Yoga", "duration_min": 60, "priority": "medium", "earliest_start": "18:00", "latest_end": "19:00", "daily_repeat": True}
                ]
            }
        },
        {
            "desc": "Classes Wed/Fri, project mornings daily",
            "prompt": "Classes on Wednesday and Friday from 10:00 to 14:00. Work on project 3 hours every morning.",
            "json": {
                "day_start": "07:00",
                "day_end": "22:00",
                "weekly_fixed": {
                    "wed": [{"label": "Classes", "start": "10:00", "end": "14:00"}],
                    "fri": [{"label": "Classes", "start": "10:00", "end": "14:00"}]
                },
                "tasks": [
                    {"title": "Project work", "duration_min": 180, "priority": "high", "earliest_start": "07:00", "latest_end": "12:00", "daily_repeat": True}
                ]
            }
        },
        {
            "desc": "Weekend hiking, weekday training, nightly reading",
            "prompt": "Hiking on weekends 08:00-11:00. Strength training on weekdays 18:00-19:30. Read every night 21:00-22:00.",
            "json": {
                "day_start": "06:00",
                "day_end": "22:30",
                "weekly_fixed": {
                    "sat": [{"label": "Hiking", "start": "08:00", "end": "11:00"}],
                    "sun": [{"label": "Hiking", "start": "08:00", "end": "11:00"}]
                },
                "tasks": [
                    {"title": "Strength training", "duration_min": 90, "priority": "medium", "earliest_start": "18:00", "latest_end": "19:30", "daily_repeat": True},
                    {"title": "Read", "duration_min": 60, "priority": "low", "earliest_start": "21:00", "latest_end": "22:00", "daily_repeat": True}
                ]
            }
        }
    ]


RAG_EXAMPLES = load_rag_examples()


def _embed(text: str) -> dict[str, int]:
    tokens = re.findall(r"[a-z]+", text.lower())
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rag_retrieve(user_request: str, k: int = 3) -> list[dict]:
    """Lightweight embedding retrieval over editable examples."""
    if not user_request:
        return []
    query_emb = _embed(user_request)
    scored = []
    for ex in RAG_EXAMPLES:
        score = _cosine(query_emb, _embed(ex.get("prompt", "")))
        scored.append((score, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for score, ex in scored[:k] if score > 0]
# -----------------------------
# Sanitizers + Helpers
# -----------------------------
def sanitize_extracted_json(data: dict) -> dict:
    """Fix common LLM issues: null ints, missing keys, placeholders."""
    if not isinstance(data, dict):
        return {}

    # day_start/day_end
    if data.get("day_start") in (None, "", "HH:MM"):
        data["day_start"] = "06:00"
    if data.get("day_end") in (None, "", "HH:MM"):
        data["day_end"] = "22:00"

    # commute ints
    if data.get("commute_before_min") is None:
        data["commute_before_min"] = 0
    if data.get("commute_after_min") is None:
        data["commute_after_min"] = 0

    # weekly_fixed existence
    wf = data.get("weekly_fixed")
    if wf is None or not isinstance(wf, dict):
        wf = {}
    for d in DAYS:
        if d not in wf or wf[d] is None:
            wf[d] = []
    data["weekly_fixed"] = wf

    # tasks existence and durations
    tasks = data.get("tasks")
    if tasks is None or not isinstance(tasks, list):
        tasks = []
    for t in tasks:
        if isinstance(t, dict):
            if t.get("duration_min") is None:
                t["duration_min"] = 60
            if t.get("priority") is None:
                t["priority"] = "medium"
            if t.get("title") is None:
                t["title"] = "Task"
    data["tasks"] = tasks

    # assumptions
    if data.get("assumptions") is None:
        data["assumptions"] = []

    return data


def parse_llm_json(raw: str) -> dict:
    """
    Robust JSON extraction: tolerate code fences or surrounding text.
    """
    if not isinstance(raw, str):
        return {}

    cleaned = raw.strip()

    # strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # after stripping backticks, remove possible `json` hint
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    # try full parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # fallback: extract first JSON object substring
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    return {}


def derive_from_prompt(user_request: str, notes: list) -> dict:
    """
    Deterministic parser: extract only what the user stated (no defaults).
    Supports:
    - "<activity> on Monday and Wednesday 09:00-11:00"
    - "<activity> on weekdays 09:00-17:00"
    - "<activity> every day 17:00-18:30"
    - "<activity>: Monday and Saturday, 06:30-07:30"
    - "<activity>, 2 hours ... between 17:00-21:00" (assumed daily)
    """
    raw_text = user_request or ""
    # Normalize common separators and bullet markers
    text = raw_text.replace("→", "->")
    weekly_fixed = {d: [] for d in DAYS}
    tasks = []
    starts = []
    ends = []
    day_start_hint = None
    day_end_hint = None
    lower_full = text.lower()
    global_daily = "every day" in lower_full or re.search(r"\bdaily\b", lower_full) is not None

    day_aliases = DAY_ALIASES

    def clean_label(label: str) -> str:
        lbl = (label or "").replace("ONLY", "").replace("only", "")
        for d in DAY_ALIASES:
            lbl = re.sub(rf"\b{d}\b", "", lbl, flags=re.IGNORECASE)
        lbl = re.sub(r"\band\b", "", lbl, flags=re.IGNORECASE)
        lbl = re.sub(r"\s+", " ", lbl).strip()
        return lbl or "Task"

    def has_title(lbl: str) -> bool:
        return any((t.get("title") or "").lower() == lbl.lower() for t in tasks)

    def normalize_days(raw: str) -> list[str]:
        raw_lower = raw.lower()
        if "weekday" in raw_lower:
            return ["mon", "tue", "wed", "thu", "fri"]
        if "weekend" in raw_lower:
            return ["sat", "sun"]
        tokens = re.split(r"[,&\s]+", raw)
        mapped = [day_aliases.get(t.lower()) for t in tokens if day_aliases.get(t.lower())]
        return mapped

    # Day window hint: "day is 06:00–22:00" or separate start/end lines
    day_window = re.search(r"day\s+is\s+(\d{1,2}:\d{2})\s*[–\-to]+\s*(\d{1,2}:\d{2})", text, re.IGNORECASE)
    if not day_window:
        day_start_match = re.search(r"day\s+starts?\s+at\s+(\d{1,2}:\d{2})", text, re.IGNORECASE)
        day_end_match = re.search(r"day\s+ends?\s+at\s+(\d{1,2}:\d{2})", text, re.IGNORECASE)
        if day_start_match and day_end_match:
            day_start_hint, day_end_hint = day_start_match.group(1), day_end_match.group(1)
    else:
        day_start_hint, day_end_hint = day_window.group(1), day_window.group(2)
    if day_start_hint:
        starts.append(day_start_hint)
    if day_end_hint:
        ends.append(day_end_hint)

    # helper to avoid duplicate tasks
    def add_task_unique(task: dict):
        title = clean_label(task.get("title", ""))
        task = {**task, "title": title}
        start = task.get("earliest_start")
        end = task.get("latest_end")
        dur = task.get("duration_min")
        for idx, t in enumerate(tasks):
            if (t.get("title") or "").lower() == title and t.get("duration_min") == dur:
                t_start = t.get("earliest_start")
                t_end = t.get("latest_end")
                if start and end and (not t_start or not t_end):
                    tasks[idx] = {**t, "earliest_start": start, "latest_end": end}
                return
        tasks.append(task)

    # Pattern: "<label> on mon, wed 09:00-11:00" or "on weekdays"
    pattern_on_days = re.compile(r"([A-Za-z][A-Za-z\s]+?)\s+on\s+([A-Za-z,\s&]+)\s+(\d{1,2}:\d{2})\s*[-to–]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_on_days.finditer(text):
        label = clean_label(m.group(1).strip())
        days_raw = m.group(2)
        start = m.group(3)
        end = m.group(4)
        mapped_days = normalize_days(days_raw)
        for d in mapped_days:
            weekly_fixed[d].append({"label": label, "start": start, "end": end, "type": "fixed"})
            starts.append(start)
            ends.append(end)

    # Pattern: "<label> -> Monday and Saturday -> 06:30-07:30"
    pattern_arrow_days = re.compile(r"([A-Za-z][A-Za-z\s]+?)\s*->\s*([A-Za-z,\s&]+)\s*->\s*(\d{1,2}:\d{2})\s*[–\-to]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_arrow_days.finditer(text):
        label = clean_label(m.group(1).strip())
        days_raw = m.group(2)
        start = m.group(3)
        end = m.group(4)
        mapped_days = normalize_days(days_raw)
        for d in mapped_days:
            weekly_fixed[d].append({"label": label, "start": start, "end": end, "type": "fixed"})
            starts.append(start)
            ends.append(end)

    # Pattern: "[ ] Label -> Monday, Saturday -> 06:30-07:30"
    pattern_arrow_fixed = re.compile(r"\[\s*\]\s*([A-Za-z][A-Za-z\s\(\)/]+?)\s*->\s*([A-Za-z,\s&/]+?)\s*->\s*(\d{1,2}:\d{2})\s*[–\-to]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_arrow_fixed.finditer(text):
        label = clean_label(m.group(1).strip())
        days_raw = m.group(2)
        start = m.group(3)
        end = m.group(4)
        mapped_days = normalize_days(days_raw)
        for d in mapped_days:
            weekly_fixed[d].append({"label": label, "start": start, "end": end, "type": "fixed"})
            starts.append(start)
            ends.append(end)

    # Pattern: "<label>: Monday and Saturday, 06:30-07:30"
    pattern_colon_days = re.compile(r"([A-Za-z][A-Za-z\s]+?):\s*([A-Za-z,\s&]+)\s*,\s*(\d{1,2}:\d{2})\s*[-to–]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_colon_days.finditer(text):
        label = clean_label(m.group(1).strip())
        days_raw = m.group(2)
        start = m.group(3)
        end = m.group(4)
        mapped_days = normalize_days(days_raw)
        for d in mapped_days:
            weekly_fixed[d].append({"label": label, "start": start, "end": end, "type": "fixed"})
            starts.append(start)
            ends.append(end)

    # Pattern: "<label> every day 17:00-18:30"
    pattern_every_day = re.compile(r"([A-Za-z][A-Za-z\s]+?)\s+(?:every\s*day|everyday|daily)\s+(\d{1,2}:\d{2})\s*[-to–]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_every_day.finditer(text):
        label = clean_label(m.group(1).strip())
        start = m.group(2)
        end = m.group(3)
        dur = max(5, to_min(end) - to_min(start))
        add_task_unique({
            "title": label,
            "duration_min": dur,
            "earliest_start": start,
            "latest_end": end,
            "priority": "medium",
            "daily_repeat": True
        })
        starts.append(start)
        ends.append(end)

    # Pattern: "<label>, 2 hours ... between 17:00-21:00" (assume daily)
    pattern_between = re.compile(r"([A-Za-z][A-Za-z\s]+?),.*?(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minutes|min).*?between\s+(\d{1,2}:\d{2})\s*[-to–]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_between.finditer(text):
        label = clean_label(m.group(1).strip())
        val = float(m.group(2))
        unit = m.group(3).lower()
        start = m.group(4)
        end = m.group(5)
        dur = int(val * 60) if "hour" in unit or "hr" in unit else int(val)
        add_task_unique({
            "title": label,
            "duration_min": dur,
            "earliest_start": start,
            "latest_end": end,
            "priority": "medium",
            "daily_repeat": True
        })
        starts.append(start)
        ends.append(end)

    # Pattern: "<label>, 90 minutes" or "<label>, 2 hours" (no time window; allow arrows/commas)
    pattern_duration_only = re.compile(r"([A-Za-z][A-Za-z\s]+?)[,:\-]\s*(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minutes|min)(?!.*between)", re.IGNORECASE)
    for m in pattern_duration_only.finditer(text):
        label = clean_label(m.group(1).strip())
        val = float(m.group(2))
        unit = m.group(3).lower()
        dur = int(val * 60) if "hour" in unit or "hr" in unit else int(val)
        add_task_unique({
            "title": label,
            "duration_min": dur,
            "priority": "medium",
            "daily_repeat": True
        })

    # Pattern: "[ ] Label -> 2 hours -> between 17:00-21:00"
    pattern_arrow_duration_window = re.compile(r"\[\s*\]\s*([A-Za-z][A-Za-z\s\(\)/]+?)\s*->\s*(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minutes|min).*?(\d{1,2}:\d{2})\s*[–\-to]+\s*(\d{1,2}:\d{2})", re.IGNORECASE)
    for m in pattern_arrow_duration_window.finditer(text):
        label = clean_label(m.group(1).strip())
        val = float(m.group(2))
        unit = m.group(3).lower()
        start = m.group(4)
        end = m.group(5)
        dur = int(val * 60) if "hour" in unit or "hr" in unit else int(val)
        add_task_unique({
            "title": label,
            "duration_min": dur,
            "earliest_start": start,
            "latest_end": end,
            "priority": "high" if "high" in label.lower() else "medium",
            "daily_repeat": True
        })
        starts.append(start)
        ends.append(end)

    # Pattern: "[ ] Label -> 90 minutes -> high priority" (no time window)
    pattern_arrow_duration_only = re.compile(r"\[\s*\]\s*([A-Za-z][A-Za-z\s\(\)/]+?)\s*->\s*(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minutes|min)\s*(?:->\s*(high|medium|low))?", re.IGNORECASE)
    for m in pattern_arrow_duration_only.finditer(text):
        label = clean_label(m.group(1).strip())
        val = float(m.group(2))
        unit = m.group(3).lower()
        pr = m.group(4).lower() if m.group(4) else "medium"
        dur = int(val * 60) if "hour" in unit or "hr" in unit else int(val)
        add_task_unique({
            "title": label,
            "duration_min": dur,
            "priority": pr,
            "daily_repeat": True
        })

    # Pattern: "<label> ... 17:00-21:00" general time window (assume daily if global_daily)
    pattern_general_window = re.compile(r"([A-Za-z][A-Za-z\s]+?)\s*\(?\s*(\d{1,2}:\d{2})\s*[–\-to]+\s*(\d{1,2}:\d{2})\s*\)?", re.IGNORECASE)
    for m in pattern_general_window.finditer(text):
        label = clean_label(m.group(1).strip())
        start = m.group(2)
        end = m.group(3)
        dur = max(5, to_min(end) - to_min(start))
        add_task_unique({
            "title": label,
            "duration_min": dur,
            "earliest_start": start,
            "latest_end": end,
            "priority": "medium",
            "daily_repeat": True if "every day" in lower_full else False
        })
        starts.append(start)
        ends.append(end)

    earliest = min(starts, key=to_min) if starts else None
    latest = max(ends, key=to_min) if ends else None

    # Deduplicate fixed events
    for d in DAYS:
        seen = set()
        unique = []
        for ev in weekly_fixed.get(d, []):
            key = ( (ev.get("label") or "").lower(), ev.get("start"), ev.get("end") )
            if key in seen:
                continue
            seen.add(key)
            unique.append(ev)
        weekly_fixed[d] = unique

    # Filter noisy task titles
    noisy = {"evening only","between","rules","fixed events","daily tasks","no overlaps","hard constraint","and thursday","and friday","and monday","and tuesday","and wednesday","and saturday","and sunday"}
    tasks = [t for t in tasks if (t.get("title") or "").lower().strip() not in noisy]
    # Deduplicate tasks
    dedup = []
    seen_t = set()
    for t in tasks:
        key = ((t.get("title") or "").lower(), int(t.get("duration_min",0)), t.get("earliest_start"), t.get("latest_end"))
        if key in seen_t:
            continue
        seen_t.add(key)
        dedup.append(t)
    tasks = dedup

    # Fill missing windows for daily tasks using day hints
    for i, t in enumerate(tasks):
        if t.get("daily_repeat") and (t.get("earliest_start") is None or t.get("latest_end") is None):
            tasks[i] = {
                **t,
                "earliest_start": t.get("earliest_start") or (day_start_hint or "06:00"),
                "latest_end": t.get("latest_end") or (day_end_hint or "22:00")
            }

    notes.append("Applied deterministic parse from prompt only (no defaults).")
    return {
        "day_start": earliest or day_start_hint or "06:00",
        "day_end": latest or day_end_hint or "22:00",
        "weekly_fixed": weekly_fixed,
        "tasks": tasks
    }


def fallback_from_prompt(user_request: str, extracted: dict, notes: list) -> dict:
    """
    Heuristic parser to recover from empty LLM outputs.
    """
    text = (user_request or "").lower()

    day_start = extracted.get("day_start") or ("06:00" if "6am" in text or "6 am" in text else "06:00")
    day_end = extracted.get("day_end") or ("22:00" if "10pm" in text or "10 pm" in text else "22:00")

    weekly_fixed = normalize_weekly_fixed(extracted.get("weekly_fixed", {}))

    # Temple Monday/Saturday early morning
    if "temple" in text:
        for d in ["mon", "sat"]:
            weekly_fixed[d].append({"label": "Temple", "start": "06:30", "end": "07:30", "type": "fixed"})

    # University Tuesday/Thursday
    if "university" in text or "class" in text:
        for d in ["tue", "thu"]:
            weekly_fixed[d].append({"label": "University", "start": "12:00", "end": "18:00", "type": "fixed"})

    tasks = extracted.get("tasks") or []

    def add_task_if_missing(title: str, **kwargs):
        if any((t.get("title") or "").lower() == title.lower() for t in tasks):
            return
        tasks.append({"title": title, **kwargs})

    if "interview" in text:
        add_task_if_missing("Prepare for interviews", duration_min=90, priority="high", daily_repeat=True)

    if "gym" in text:
        add_task_if_missing(
            "Gym",
            duration_min=120,
            priority="medium",
            earliest_start="17:00",
            latest_end="21:00",
            daily_repeat=True,
        )

    if "dinner" in text:
        add_task_if_missing(
            "Prepare dinner",
            duration_min=60,
            priority="low",
            earliest_start="19:00",
            latest_end="21:00",
            daily_repeat=True,
        )

    if not tasks:
        tasks = [{"title": "Prepare for interviews", "duration_min": 90, "priority": "high", "daily_repeat": True}]
        notes.append("Fallback: added default interview prep task.")

    notes.append("Applied heuristic fallback from prompt.")
    return {
        "day_start": day_start,
        "day_end": day_end,
        "weekly_fixed": weekly_fixed,
        "tasks": tasks
    }


def ensure_required_items(user_request: str, weekly_fixed: Dict[str, List[Dict[str, Any]]], tasks: List[Dict[str, Any]], notes: list) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Augment LLM extraction with must-have items inferred from the prompt.
    Avoid duplicates by label/title match.
    """
    text = (user_request or "").lower()
    weekly_fixed = normalize_weekly_fixed(weekly_fixed or {})
    tasks = list(tasks or [])

    def has_event(label_sub: str, days: List[str]) -> bool:
        for d in days:
            for e in weekly_fixed.get(d, []):
                if label_sub in (e.get("label") or "").lower():
                    return True
        return False

    def has_task(title_sub: str) -> bool:
        return any(title_sub in (t.get("title") or "").lower() for t in tasks)

    def upsert_task(title: str, **attrs):
        found = False
        for i, t in enumerate(tasks):
            if title.lower() == (t.get("title") or "").lower():
                merged = {**t}
                for k, v in attrs.items():
                    if k not in merged or merged[k] in (None, ""):
                        merged[k] = v
                tasks[i] = merged
                found = True
        if not found:
            tasks.append({"title": title, **attrs})

    def mark_repeat(title_sub: str):
        updated = False
        for i, t in enumerate(tasks):
            if title_sub in (t.get("title") or "").lower():
                tasks[i] = {**t, "daily_repeat": True}
                updated = True
        return updated

    def update_task_fields(title_sub: str, **updates):
        """Apply non-null updates even if fields already exist."""
        for i, t in enumerate(tasks):
            if title_sub in (t.get("title") or "").lower():
                merged = {**t}
                for k, v in updates.items():
                    if v is not None:
                        merged[k] = v
                tasks[i] = merged

    # Temple on Monday/Saturday early morning
    if "temple" in text and not has_event("temple", ["mon", "sat"]):
        for d in ["mon", "sat"]:
            weekly_fixed[d].append({"label": "Temple", "start": "06:30", "end": "07:30", "type": "fixed"})
        notes.append("Added temple blocks to Mon/Sat from prompt.")

    # University/College on Tuesday/Thursday
    if (any(k in text for k in ["university", "college", "class", "lecture", "course"])) and not has_event("university", ["tue", "thu"]):
        uni_start = None
        uni_end = None
        time_match = re.findall(r"(\d{1,2}:\d{2})", text)
        if len(time_match) >= 2:
            uni_start, uni_end = time_match[0], time_match[1]
        elif "10:00" in text or "10am" in text or "10 am" in text:
            uni_start, uni_end = "10:00", "18:00"
        if not uni_start:
            uni_start = "12:00"
        if not uni_end:
            uni_end = "18:00"
        for d in ["tue", "thu"]:
            weekly_fixed[d].append({"label": "University", "start": uni_start, "end": uni_end, "type": "fixed"})
        notes.append(f"Added university blocks to Tue/Thu from prompt ({uni_start}-{uni_end}).")

    # Interview prep daily
    if "interview" in text:
        if not mark_repeat("interview"):
            tasks.append({"title": "Prepare for interviews", "duration_min": 90, "priority": "high", "daily_repeat": True})
            notes.append("Added interview prep task from prompt.")
        else:
            update_task_fields("interview", duration_min=90, priority="high")
            notes.append("Marked interview prep as daily_repeat from prompt.")

    # Gym daily evening (respect duration hints)
    if "gym" in text:
        gym_dur = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minutes|min)", text)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            gym_dur = int(val * 60) if "hour" in unit or "hr" in unit else int(val)
        if mark_repeat("gym"):
            update_task_fields("gym", duration_min=gym_dur, priority="medium", earliest_start="17:00", latest_end="21:00")
            notes.append(f"Marked gym as daily_repeat (using prompt duration {gym_dur or 'existing'}).")
        else:
            upsert_task(
                "Gym",
                duration_min=gym_dur or 120,
                priority="medium",
                earliest_start="17:00",
                latest_end="21:00",
                daily_repeat=True
            )
            notes.append(f"Added daily gym task ({gym_dur or 120} min) from prompt.")

    # Dinner daily evening
    if "dinner" in text:
        dinner_dur = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minutes|min)", text)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            dinner_dur = int(val * 60) if "hour" in unit or "hr" in unit else int(val)
        if mark_repeat("dinner"):
            update_task_fields("dinner", duration_min=dinner_dur, priority="low", earliest_start="19:00", latest_end="21:00")
            notes.append(f"Marked dinner prep as daily_repeat (using prompt duration {dinner_dur or 'existing'}).")
        else:
            upsert_task(
                "Prepare dinner",
                duration_min=dinner_dur or 60,
                priority="low",
                earliest_start="19:00",
                latest_end="21:00",
                daily_repeat=True
            )
            notes.append(f"Added daily dinner prep task ({dinner_dur or 60} min) from prompt.")

    return weekly_fixed, tasks


def normalize_weekly_fixed(weekly_fixed: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Ensure all 7 day keys exist and are lists."""
    out = {d: [] for d in DAYS}
    if not isinstance(weekly_fixed, dict):
        return out
    for d in DAYS:
        v = weekly_fixed.get(d, [])
        out[d] = v if isinstance(v, list) else []
    return out


def dedup_fixed_events(weekly_fixed: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Remove duplicate fixed events per day based on label/start/end."""
    weekly_fixed = normalize_weekly_fixed(weekly_fixed)
    for d in DAYS:
        seen = set()
        unique = []
        for ev in weekly_fixed.get(d, []):
            key = ((ev.get("label") or "").lower(), ev.get("start"), ev.get("end"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(ev)
        weekly_fixed[d] = unique
    return weekly_fixed


def find_slot(blocks: List[Dict[str, Any]], duration_min: int, earliest: str, latest_end: str, day_end: str) -> tuple[str, str] | None:
    """
    Find the first non-overlapping slot that fits within [earliest, latest_end].
    """
    start_m = max(to_min(earliest), 0)
    latest_end_m = min(to_min(latest_end), to_min(day_end))
    duration = duration_min

    # sort existing blocks
    existing = sorted(blocks, key=lambda b: to_min(b["start"]))

    cursor = start_m
    while cursor + duration <= latest_end_m:
        # check overlap
        overlap = False
        for b in existing:
            s = to_min(b["start"])
            e = to_min(b["end"])
            if not (cursor + duration <= s or cursor >= e):
                overlap = True
                cursor = e  # jump to end of overlapping block
                break
        if not overlap:
            return to_hhmm(cursor), to_hhmm(cursor + duration)
        # small nudge forward
        cursor += 5
    return None


def is_valid_hhmm(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip()
    if x.upper() == "HH:MM":
        return False
    m = re.match(r"^(\d{1,2}):(\d{2})$", x)
    if not m:
        return False
    hh = int(m.group(1))
    mm = int(m.group(2))
    return 0 <= hh <= 23 and 0 <= mm <= 59


def enforce_day_rules(user_request: str, weekly_fixed: dict, notes: list) -> dict:
    """
    Keep events only on the days mentioned IN THE SAME SENTENCE as the keyword.
    This prevents temple/university from spreading to other days just because those day names
    appear elsewhere in the prompt.
    """
    text = (user_request or "").lower()

    # split into sentences/lines
    parts = re.split(r"[.\n;]+", text)

    def days_in_fragment(fragment: str) -> set[str]:
        days = set()
        for k, v in DAY_ALIASES.items():
            if re.search(rf"\b{k}\b", fragment):
                days.add(v)
        return days

    def mentioned_days_for(keyword: str) -> set[str]:
        out = set()
        for frag in parts:
            if keyword in frag:
                out |= days_in_fragment(frag)
        return out

    def label_matches(ev_label: str, target: str) -> bool:
        return target in (ev_label or "").lower()

    weekly_fixed = normalize_weekly_fixed(weekly_fixed)

    temple_days = mentioned_days_for("temple")
    uni_days = mentioned_days_for("university")

    if temple_days:
        for d in DAYS:
            if d not in temple_days:
                weekly_fixed[d] = [e for e in weekly_fixed[d] if not label_matches(e.get("label", ""), "temple")]
        notes.append(f"Enforced: temple only on {sorted(list(temple_days))}")

    if uni_days:
        for d in DAYS:
            if d not in uni_days:
                weekly_fixed[d] = [e for e in weekly_fixed[d] if not label_matches(e.get("label", ""), "university")]
        notes.append(f"Enforced: university only on {sorted(list(uni_days))}")

    return weekly_fixed


def force_evening(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce evening-only placement for Gym.
    Evening window: 17:00–21:00
    """
    title = (task.get("title") or "").lower()
    if "gym" == title or "workout" in title:
        task["earliest_start"] = "17:00"
        task["latest_end"] = "21:00"
    return task


def add_sleep_fixed_events(weekly_fixed: Dict[str, List[Dict[str, Any]]], notes: List[str], day_end: str) -> Dict[str, List[Dict[str, Any]]]:
    """Add Sleep as a hard fixed event from day_end to 23:59 for every day."""
    weekly_fixed = normalize_weekly_fixed(weekly_fixed)

    sleep_start = day_end  # e.g., 22:00
    sleep_end = "23:59"

    if not is_valid_hhmm(sleep_start):
        sleep_start = "22:00"

    for d in DAYS:
        weekly_fixed[d].append({
            "label": "Sleep",
            "start": sleep_start,
            "end": sleep_end,
            "type": "fixed"
        })

    notes.append(f"Added hard sleep block daily: {sleep_start}–{sleep_end}")
    return weekly_fixed


def dedup_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by label/start/end per day."""
    seen = set()
    out = []
    for b in blocks:
        key = ((b.get("label") or "").lower(), b.get("start"), b.get("end"))
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


# -----------------------------
# Nodes
# -----------------------------
def node_load_saved_prefs(state: SchedulerState) -> SchedulerState:
    notes = state.get("notes", [])
    user_id = state.get("user_id", "default")

    try:
        saved = load_prefs(user_id) or {}
    except Exception as e:
        saved = {}
        notes.append(f"WARNING: load_prefs failed: {e}")

    extracted = state.get("extracted", {})
    extracted["saved_prefs"] = saved

    notes.append("Loaded saved preferences (available, but current prompt will override).")
    return {**state, "extracted": extracted, "notes": notes}


def node_extract_weekly_llm(state: SchedulerState) -> SchedulerState:
    notes = state.get("notes", [])
    user_request = state.get("user_request", "")
    user_id = state.get("user_id", "default")

    notes.append(f"DEBUG user_request: {user_request[:120]}")

    # RAG retrieval (static examples)
    retrieved = rag_retrieve(user_request, k=2)
    rag_context = ""
    if retrieved:
        parts = []
        for ex in retrieved:
            parts.append(f"- Example: {ex['desc']}\nPROMPT: {ex['prompt']}\nJSON: {json.dumps(ex['json'])}")
        rag_context = "\n\nUse these examples to mirror structure and day mappings:\n" + "\n\n".join(parts)

    try:
        raw = OLLAMA_LLM.invoke(f"{EXTRACT_PROMPT}{rag_context}\n\nUSER REQUEST:\n{user_request}").content
        data = parse_llm_json(raw)
        if not data:
            raise ValueError("empty parsed data")
    except Exception as e:
        notes.append(f"WARNING: LLM JSON parse failed; falling back to deterministic parser. {e}")
        data = {}
    data = sanitize_extracted_json(data)
    extracted = ExtractedWeekly.model_validate(data)

    # ✅ Current prompt wins (no saved fallback here)
    day_start = extracted.day_start
    day_end = extracted.day_end
    commute_before = extracted.commute_before_min
    commute_after = extracted.commute_after_min

    weekly_fixed = normalize_weekly_fixed(
        {d: [e.model_dump() for e in extracted.weekly_fixed.get(d, [])] for d in DAYS}
    )
    tasks = [t.model_dump() for t in extracted.tasks]

    # Deterministic parse to supplement LLM output (no defaults)
    derived = derive_from_prompt(user_request, notes)
    if any(derived["weekly_fixed"].values()) or derived["tasks"]:
        # merge fixed events
        for d in DAYS:
            weekly_fixed.setdefault(d, [])
            weekly_fixed[d].extend(derived["weekly_fixed"].get(d, []))
        # merge tasks
        tasks = tasks + derived["tasks"]
        # merge day bounds if provided
        if derived["day_start"]:
            day_start = derived["day_start"]
        if derived["day_end"]:
            day_end = derived["day_end"]

    # If still nothing, fall back to prompt-derived only
    if (not tasks) and all(not v for v in weekly_fixed.values()):
        derived = derive_from_prompt(user_request, notes)
        weekly_fixed = normalize_weekly_fixed(derived.get("weekly_fixed", {}))
        tasks = derived.get("tasks", [])
        if derived.get("day_start"):
            day_start = derived["day_start"]
        if derived.get("day_end"):
            day_end = derived["day_end"]

    # Deduplicate fixed events after all merges
    weekly_fixed = dedup_fixed_events(weekly_fixed)

    # 🧹 Drop invalid fixed events (bad time formats)
    for d in DAYS:
        cleaned = []
        for e in weekly_fixed.get(d, []):
            s = e.get("start")
            en = e.get("end")
            if not (is_valid_hhmm(s) and is_valid_hhmm(en)):
                notes.append(f"WARNING: Dropped invalid event on {d}: {e}")
                continue
            cleaned.append(e)
        weekly_fixed[d] = cleaned

    # ✅ Enforce day-specific rules based on user prompt (prevents repetition)
    weekly_fixed = enforce_day_rules(user_request, weekly_fixed, notes)

    # 🌙 Add hard sleep constraint
    weekly_fixed = add_sleep_fixed_events(weekly_fixed, notes, day_end)

    save_prefs(user_id, {
        "day_start": day_start,
        "day_end": day_end,
        "weekly_fixed": weekly_fixed,
        "tasks": tasks
    })

    if not tasks:
        notes.append("No tasks provided in prompt; schedule will only include fixed events if any.")

    # Save for next time (does NOT override current run)
    save_prefs(user_id, {
        "day_start": day_start,
        "day_end": day_end,
        "commute_before_min": commute_before,
        "commute_after_min": commute_after,
        "weekly_fixed": weekly_fixed,
        "tasks": tasks
    })
    notes.append("Saved latest preferences to SQLite.")

    if extracted.assumptions:
        notes.append("Assumptions: " + "; ".join(extracted.assumptions))

    return {
        **state,
        "day_start": day_start,
        "day_end": day_end,
        "commute_before_min": commute_before,
        "commute_after_min": commute_after,
        "weekly_fixed": weekly_fixed,
        "tasks": tasks,
        "notes": notes
    }


def node_build_weekly_schedule(state: SchedulerState) -> SchedulerState:
    notes = state.get("notes", [])
    weekly = {}

    day_start = state.get("day_start", "06:00")
    day_end = state.get("day_end", "22:00")

    # Prepare tasks
    base_tasks = promote_deep_work(state.get("tasks", []))
    base_tasks = [force_evening(t) for t in base_tasks]
    base_tasks = [{**t, "task_id": t.get("task_id", f"task-{i}")} for i, t in enumerate(base_tasks)]

    repeat_tasks = [t for t in base_tasks if t.get("daily_repeat")]
    once_tasks = [t for t in base_tasks if not t.get("daily_repeat")]
    once_per_day = max(1, math.ceil(len(once_tasks) / len(DAYS))) if once_tasks else 0

    def task_sort_key(t: Dict[str, Any]) -> tuple[int, int]:
        pr_map = {"high": 0, "medium": 1, "low": 2}
        earliest = to_min(t.get("earliest_start", day_start))
        return (earliest, pr_map.get(t.get("priority", "medium"), 2))

    repeat_tasks.sort(key=task_sort_key)
    once_tasks.sort(key=task_sort_key)

    # Compute dates for the coming week (starting next Monday)
    today = date.today()
    week_start = today + timedelta(days=(7 - today.weekday()) % 7)

    for offset, d in enumerate(DAYS):
        # fixed events for this day (commute disabled per requirements)
        events = list(state.get("weekly_fixed", {}).get(d, []))
        # ensure sleep block exists (defensive)
        has_sleep = any((e.get("label") or "").lower() == "sleep" for e in events)
        if not has_sleep:
            events.append({
                "label": "Sleep",
                "start": day_end,
                "end": "23:59",
                "type": "fixed"
            })

        # Build today's task list:
        today_candidates = []

        # Daily repeat tasks (clone per day)
        for t in repeat_tasks:
            today_candidates.append({**t, "task_id": f"{t.get('task_id', t.get('title','task'))}-{offset}"})

        # Spread one-time tasks across the week
        if once_tasks and once_per_day:
            start_idx = offset * once_per_day
            today_candidates.extend(once_tasks[start_idx:start_idx + once_per_day])

        daily_tasks = list(today_candidates)

        constraints = {
            "day_start": day_start,
            "day_end": day_end,
            "fixed_events": events
        }

        draft = build_draft_schedule(constraints, daily_tasks)

        # resolve overlaps a few rounds
        it = 0
        while it < 2:
            conflicts = check_conflicts(draft)
            if not conflicts:
                break
            draft = auto_fix_conflicts(draft)
            it += 1

        # final clamp
        draft = clamp_to_day_end(draft, day_end)

        # dedup same label/start/end on this day
        draft = dedup_blocks(draft)

        # Ensure all repeat tasks landed; if missing, force place within their windows
        for rt in repeat_tasks:
            title = rt.get("title", "")
            if any(title.lower() == (b.get("label") or "").lower() for b in draft):
                continue
            duration = int(rt.get("duration_min", 60))
            earliest = rt.get("earliest_start", day_start)
            latest_end = rt.get("latest_end", day_end)
            slot = find_slot(draft, duration, earliest, latest_end, day_end)
            if slot:
                s, e = slot
                draft.append({
                    "label": title,
                    "start": s,
                    "end": e,
                    "type": "task",
                    "priority": rt.get("priority", "medium"),
                    "task_id": f"{rt.get('task_id', title)}-{offset}"
                })

        # attach calendar date to each block
        day_date = (week_start + timedelta(days=offset)).isoformat()
        draft = [{**b, "date": day_date} for b in draft]

        weekly[d] = draft

    notes.append("Built weekly schedule with hard day rules + sleep + evening gym (one-week allocation).")
    return {**state, "weekly_schedule": weekly, "notes": notes}


# -----------------------------
# Graph builder
# -----------------------------
def build_graph():
    g = StateGraph(SchedulerState)

    g.add_node("load_prefs", node_load_saved_prefs)
    g.add_node("extract", node_extract_weekly_llm)
    g.add_node("schedule", node_build_weekly_schedule)

    g.set_entry_point("load_prefs")
    g.add_edge("load_prefs", "extract")
    g.add_edge("extract", "schedule")
    g.add_edge("schedule", END)

    return g.compile()

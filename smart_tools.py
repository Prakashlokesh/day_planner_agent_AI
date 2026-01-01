from typing import List, Dict
from tools import to_min, to_hhmm

def inject_breaks(tasks: List[Dict], break_every_min: int = 90, break_len_min: int = 10) -> List[Dict]:
    """
    Insert short breaks after long focus chunks.
    """
    out = []
    acc = 0
    break_idx = 0
    for t in tasks:
        out.append(t)
        acc += int(t.get("duration_min", 0))
        if acc >= break_every_min:
            out.append({
                "title": "Break / Water",
                "duration_min": break_len_min,
                "priority": "low",
                "task_id": f"break-{break_idx}"
            })
            break_idx += 1
            acc = 0
    return out

def promote_deep_work(tasks: List[Dict]) -> List[Dict]:
    """
    Ensure deep work blocks are at least 90 minutes.
    """
    out = []
    for t in tasks:
        title = t.get("title", "").lower()
        dur = int(t.get("duration_min", 60))
        if ("ai" in title or "project" in title or "chatbot" in title) and dur < 90:
            t = {**t, "duration_min": 90, "priority": "high"}
        out.append(t)
    return out

def clamp_to_day_end(blocks: List[Dict], day_end: str) -> List[Dict]:
    """
    Trim or remove blocks that exceed day_end.
    """
    end_m = to_min(day_end)
    out = []

    for b in blocks:
        s = to_min(b["start"])
        e = to_min(b["end"])

        if s >= end_m:
            continue

        if e > end_m:
            b = {**b, "end": to_hhmm(end_m)}

        out.append(b)

    return out

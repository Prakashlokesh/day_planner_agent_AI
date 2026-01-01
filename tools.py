from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
@dataclass
class Block:
    start: int  # minutes from 00:00
    end: int
    label: str

import re

def to_min(hhmm: str) -> int:
    """
    Convert 'HH:MM' to minutes since midnight.
    If invalid (e.g., 'HH:MM', None), return 0.
    """
    if not isinstance(hhmm, str):
        return 0

    hhmm = hhmm.strip()

    # Reject placeholders like "HH:MM"
    if hhmm.upper() == "HH:MM":
        return 0

    # Must match 24-hour HH:MM
    m = re.match(r"^(\d{1,2}):(\d{2})$", hhmm)
    if not m:
        return 0

    h = int(m.group(1))
    minute = int(m.group(2))

    if h < 0 or h > 23 or minute < 0 or minute > 59:
        return 0

    return h * 60 + minute


def to_hhmm(m: int) -> str:
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def check_conflicts(blocks: List[Dict]) -> List[Tuple[int, int, str, str]]:
    """Return list of overlaps: (i, j, label_i, label_j) after sorting by start time."""
    parsed = [Block(to_min(b["start"]), to_min(b["end"]), b.get("label", "Unnamed")) for b in blocks]
    parsed.sort(key=lambda x: x.start)

    overlaps = []
    for i in range(len(parsed) - 1):
        a, b = parsed[i], parsed[i + 1]
        if b.start < a.end:
            overlaps.append((i, i + 1, a.label, b.label))
    return overlaps

def shift_block(block: Dict, delta_min: int) -> Dict:
    s = to_min(block["start"]) + delta_min
    e = to_min(block["end"]) + delta_min
    # clamp at 00:00 min for safety
    s = max(0, s)
    e = max(0, e)
    return {**block, "start": to_hhmm(s), "end": to_hhmm(e)}

def auto_fix_conflicts(blocks: List[Dict], max_iters: int = 10) -> List[Dict]:
    """
    Simple heuristic:
    - sort by start
    - if overlap, push the later block forward so it starts at the earlier block's end
    """
    fixed = [dict(b) for b in blocks]

    for _ in range(max_iters):
        fixed.sort(key=lambda b: to_min(b["start"]))
        overlaps = check_conflicts(fixed)
        if not overlaps:
            return fixed

        i, j, _, _ = overlaps[0]
        a, b = fixed[i], fixed[j]
        push_to = to_min(a["end"])
        delta = push_to - to_min(b["start"])
        fixed[j] = shift_block(b, delta)

    return fixed

def build_draft_schedule(constraints, tasks):
    day_start = constraints["day_start"]
    day_end = constraints["day_end"]
    fixed_events = constraints.get("fixed_events", [])

    blocks = []
    # Add fixed events first
    for e in fixed_events:
        blocks.append({
            "label": e.get("label", "Fixed"),
            "start": e["start"],
            "end": e["end"],
            "type": "fixed"
        })

    # sort helper
    def sorted_blocks():
        return sorted(blocks, key=lambda b: to_min(b["start"]))

    # find next free pointer (after fixed events)
    def next_free_pointer(pointer):
        for ev in sorted_blocks():
            s = to_min(ev["start"])
            e = to_min(ev["end"])
            # if pointer falls inside an existing block, jump to end
            if s <= pointer < e:
                pointer = e
        return pointer

    pointer = to_min(day_start)
    day_end_m = to_min(day_end)

    for task in tasks:
        title = task.get("title", "Task")
        dur = int(task.get("duration_min", 60))
        pr = task.get("priority", "medium")

        # ✅ NEW: time windows (optional per task)
        earliest = to_min(task.get("earliest_start", day_start))
        latest_end = to_min(task.get("latest_end", day_end))

        # ensure pointer respects earliest
        pointer = max(pointer, earliest)
        pointer = next_free_pointer(pointer)

        # if can't fit within allowed window, skip
        if pointer + dur > latest_end or pointer + dur > day_end_m:
            continue

        # if overlaps fixed blocks, move pointer until free and re-check fit
        tries = 0
        while tries < 50:
            pointer = next_free_pointer(pointer)
            if pointer + dur <= latest_end and pointer + dur <= day_end_m:
                # also ensure it doesn't start inside another block
                overlap = False
                for ev in sorted_blocks():
                    s = to_min(ev["start"])
                    e = to_min(ev["end"])
                    if not (pointer + dur <= s or pointer >= e):
                        overlap = True
                        pointer = e
                        break
                if not overlap:
                    break
            else:
                pointer += 5  # nudge forward a bit
            tries += 1

        if pointer + dur > latest_end or pointer + dur > day_end_m:
            continue

        blocks.append({
            "label": title,
            "start": to_hhmm(pointer),
            "end": to_hhmm(pointer + dur),
            "type": "task",
            "priority": pr,
            "task_id": task.get("task_id")
        })

        pointer = pointer + dur  # advance pointer after scheduling task

    return sorted_blocks()
def inject_commute(events, commute_before_min=0, commute_after_min=0):
    """
    Add commute blocks before and after fixed events.
    """
    if not commute_before_min and not commute_after_min:
        return events

    out = []
    for e in events:
        start = e["start"]
        end = e["end"]

        if commute_before_min:
            out.append({
                "label": "Commute",
                "start": to_hhmm(to_min(start) - commute_before_min),
                "end": start,
                "type": "fixed"
            })

        out.append(e)

        if commute_after_min:
            out.append({
                "label": "Commute",
                "start": end,
                "end": to_hhmm(to_min(end) + commute_after_min),
                "type": "fixed"
            })

    return out

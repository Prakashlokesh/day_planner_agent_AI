from typing import Dict, List
from datetime import datetime, timedelta, date

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_TO_OFFSET = {d: i for i, d in enumerate(DAYS)}

def _dt(d: date, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime(d.year, d.month, d.day, int(h), int(m))

def schedule_to_ics(weekly_schedule: Dict[str, List[dict]], week_start: date) -> str:
    """
    Convert weekly schedule to .ics calendar format.
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AgenticPlanner//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH"
    ]

    for day, blocks in weekly_schedule.items():
        if day not in DAY_TO_OFFSET:
            continue

        event_date = week_start + timedelta(days=DAY_TO_OFFSET[day])

        for i, b in enumerate(blocks):
            title = b.get("label", "Task")
            start = b.get("start")
            end = b.get("end")

            if not start or not end:
                continue

            dtstart = _dt(event_date, start).strftime("%Y%m%dT%H%M%S")
            dtend = _dt(event_date, end).strftime("%Y%m%dT%H%M%S")
            uid = f"{day}-{i}-{dtstart}@agenticplanner"

            lines.extend([
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}Z",
                f"DTSTART:{dtstart}",
                f"DTEND:{dtend}",
                f"SUMMARY:{title}",
                "END:VEVENT"
            ])

    lines.append("END:VCALENDAR")
    return "\n".join(lines)

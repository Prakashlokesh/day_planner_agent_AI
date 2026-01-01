import streamlit as st
from datetime import date, timedelta

from graph import build_graph
from memory import init_db
from ics_export import schedule_to_ics
import os

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_NAMES = {
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}


def render_day(day_key: str, blocks):
    day_label = DAY_NAMES.get(day_key, day_key.upper())
    # sort by start
    def to_min(hhmm):
        try:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)
        except Exception:
            return 0
    blocks = sorted(blocks or [], key=lambda b: to_min(b.get("start", "00:00")))
    date_str = blocks[0].get("date") if blocks else ""
    header = f"<div class='day-card'><div class='day-title'>{day_label} {date_str}</div>"
    if not blocks:
        header += "<div class='muted'>No events</div></div>"
        st.markdown(header, unsafe_allow_html=True)
        return
    items = []
    for b in blocks:
        label = b.get("label", "Task")
        start = b.get("start", "")
        end = b.get("end", "")
        btype = b.get("type", "task")
        pr = (b.get("priority") or "").upper()
        badge = "badge-fixed" if btype == "fixed" else "badge-task"
        pr_text = f" · {pr}" if pr else ""
        items.append(f"<div class='item'><span class='badge {badge}'>{btype.title()}</span>{pr_text} — <strong>{label}</strong> &nbsp; {start}–{end}</div>")
    html = header + "".join(items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

# Must be first Streamlit command
st.set_page_config(page_title="Agentic Day Planner", layout="centered")

# Inject lightweight styling
st.markdown(
    """
    <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --card: #1f2937;
      --accent: #22d3ee;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --badge-fixed: #60a5fa;
      --badge-task: #34d399;
    }
    body { background: linear-gradient(135deg, #0f172a 0%, #0b1120 100%); color: var(--text); }
    .planner-panel { background: var(--panel); padding: 18px; border-radius: 14px; border: 1px solid #1f2937; box-shadow: 0 10px 40px rgba(0,0,0,0.35); }
    .day-card { background: var(--card); padding: 14px 16px; border-radius: 12px; margin-bottom: 10px; border: 1px solid #1f2937; }
    .day-title { font-weight: 700; font-size: 15px; letter-spacing: 0.4px; color: var(--accent); }
    .item { margin: 6px 0; color: var(--text); font-size: 14px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; color: #0b1120; }
    .badge-fixed { background: var(--badge-fixed); }
    .badge-task { background: var(--badge-task); }
    .muted { color: var(--muted); font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("🧹 Clear saved memory"):
    if os.path.exists("planner.db"):
        os.remove("planner.db")
    st.success("Cleared memory. Please refresh the page.")

init_db()
@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()


st.title("🗓️ Agentic Scheduling Bot")
st.caption("Local LLM (Ollama) + LangGraph + Tools + Memory + Calendar export")

user_id = st.text_input("User ID", value="prakash")

if "history" not in st.session_state:
    st.session_state.history = []

msg = st.text_area("What do you want to plan? (work/class/commute/tasks)", height=120)

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Weekly Plan"):
        result = graph.invoke({"user_id": user_id, "user_request": msg})
        st.session_state.history.append(("user", msg))
        st.session_state.history.append(("bot", result.get("weekly_schedule", {})))
        st.session_state.last_result = result

with col2:
    if st.button("Clear"):
        st.session_state.history = []
        st.session_state.last_result = None

st.divider()

if st.session_state.history:
    st.subheader("Conversation")
    for role, content in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown("**Bot:** Weekly schedule generated ✅")
            weekly = content or {}
            for d in DAYS:
                blocks = weekly.get(d, [])
                render_day(d, blocks)

st.divider()

# ICS export
if getattr(st.session_state, "last_result", None):
    weekly = st.session_state.last_result.get("weekly_schedule", {})
    st.subheader("📅 Export to Google Calendar (.ics)")

    # pick next Monday
    today = date.today()
    next_monday = today + timedelta(days=(7 - today.weekday()) % 7)  # Monday=0
    week_start = st.date_input("Week start (Monday)", value=next_monday)

    ics_text = schedule_to_ics(weekly, week_start=week_start)
    st.download_button(
        "Download .ics file",
        data=ics_text,
        file_name="weekly_plan.ics",
        mime="text/calendar"
    )

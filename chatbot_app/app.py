import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _path in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st

import app_runtime as rt

SUGGESTIONS = [
    "NVIDIA (NVDA) latest stock news",
    "Apple (AAPL) market outlook",
    "US semiconductor industry news",
    "Tesla (TSLA) recent developments",
]

st.set_page_config(
    page_title="Finnessly",
    page_icon=":material/monitoring:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("current_chat_id", None)


def _render_tool_calls(tool_calls):
    for call in tool_calls:
        if call["failure"]:
            st.markdown(f"**{call['name']}** · `{call['args']}` · failed: {call['failure']}")
        else:
            st.markdown(f"**{call['name']}** · `{call['args']}`")
        st.text(call["output"])


def _render_agent_steps(agents):
    for agent in agents:
        duration = rt.format_duration(agent["duration_s"]) if agent["duration_s"] is not None else "running"
        with st.status(f"{agent['role']} — {duration}", type="step", expanded=False):
            st.caption(agent["task"] or agent["role"])
            _render_tool_calls(agent["tool_calls"])


def _assistant_label(msg):
    label = rt.WORKFLOWS[msg["workflow"]]["label"]
    duration = rt.format_duration(msg.get("total_s"))
    if msg.get("failed"):
        return f"Workflow {label} failed after {duration}"
    agents = len(msg.get("agents", []))
    return f"Workflow {label} · completed in {duration} · {agents} agent{'s' if agents != 1 else ''}"


def _render_assistant_message(msg, chat_id, index):
    with st.chat_message("assistant"):
        state = "error" if msg.get("failed") else "complete"
        with st.status(_assistant_label(msg), type="compact", state=state, expanded=False):
            if msg.get("failed"):
                st.error(msg.get("error") or "The workflow run failed.")
            else:
                _render_agent_steps(msg.get("agents", []))
        if msg.get("content"):
            st.markdown(msg["content"])
            st.download_button(
                "Download report",
                data=msg["content"],
                file_name=f"finnessly_report_{chat_id}_{index}.md",
                mime="text/markdown",
                key=f"download_{chat_id}_{index}",
            )


def _ticker_text(snap):
    parts = [f"**{rt.format_duration(snap['elapsed_s'])} elapsed**"]
    if snap["current"]:
        parts.append(f"running `{snap['current']['role']}`")
    if snap["last_tool"]:
        parts.append(f"last tool: {snap['last_tool']}")
    if snap["tool_call_count"]:
        plural = "s" if snap["tool_call_count"] != 1 else ""
        parts.append(f"{snap['tool_call_count']} tool call{plural}")
    return " · ".join(parts)


def _update_live_step(agent, steps_box, live_steps, completed):
    role = agent["role"]
    step = live_steps.get(role)
    if step is None:
        with steps_box.status(f":shimmer[{role}…]", type="step", expanded=False) as new_step:
            st.caption(agent["task"] or role)
        live_steps[role] = new_step
        step = new_step
    if agent["duration_s"] is not None and role not in completed:
        with step:
            _render_tool_calls(agent["tool_calls"])
        step.update(label=f"{role} — {rt.format_duration(agent['duration_s'])}", state="complete")
        completed.add(role)


def _run_workflow_live(chat, prompt):
    workflow_name = chat["workflow"]
    workflow_label = rt.WORKFLOWS[workflow_name]["label"]
    with st.chat_message("assistant"):
        run = rt.CrewRun(workflow_name, prompt)
        with st.status(f":shimmer[Loading workflow {workflow_label}…]", type="compact", expanded=True) as status:
            ticker = st.empty()
            steps_box = st.container()
            live_steps: dict[str, object] = {}
            completed: set[str] = set()
            run.start()
            while True:
                snap = run.recorder.snapshot()
                for agent in snap["agents"]:
                    _update_live_step(agent, steps_box, live_steps, completed)
                ticker.markdown(_ticker_text(snap))
                if snap["agents"]:
                    status.update(label=f":shimmer[Workflow {workflow_label} running — {rt.format_duration(snap['elapsed_s'])}]")
                if run.is_done:
                    break
                time.sleep(1.0)
            run.join()
            snap = run.recorder.snapshot()
            for agent in snap["agents"]:
                _update_live_step(agent, steps_box, live_steps, completed)
            message = run.build_message()
            snap["elapsed_s"] = run.total_s or 0.0
            ticker.markdown(_ticker_text(snap))
            if message["failed"]:
                status.update(
                    label=f"Workflow {workflow_label} failed after {rt.format_duration(run.total_s)}",
                    state="error",
                )
                with status:
                    st.error(message["error"] or "The workflow run failed.")
            else:
                status.update(
                    label=f"Workflow {workflow_label} completed in {rt.format_duration(run.total_s)}",
                    state="complete",
                )
            chat["messages"].append(message)
            rt.save_chat(chat)
        if message["content"]:
            st.markdown(message["content"])
            index = len(chat["messages"]) - 1
            st.download_button(
                "Download report",
                data=message["content"],
                file_name=f"finnessly_report_{chat['id']}_{index}.md",
                mime="text/markdown",
                key=f"download_live_{chat['id']}_{index}",
            )


chats = rt.list_chats()
if st.session_state.current_chat_id is None and chats:
    st.session_state.current_chat_id = chats[0]["id"]
current = next((c for c in chats if c["id"] == st.session_state.current_chat_id), None)
if st.session_state.current_chat_id is not None and current is None:
    st.session_state.current_chat_id = None

with st.sidebar:
    st.title("Finnessly")
    st.caption("Agentic stock-news analysis")

    if st.button("New chat", icon=":material/add:", use_container_width=True):
        chat = rt.new_chat()
        st.session_state.current_chat_id = chat["id"]
        st.rerun()

    st.divider()
    for chat in chats:
        col_open, col_delete = st.columns([0.8, 0.2], vertical_alignment="center")
        if col_open.button(
            chat["title"],
            key=f"open_{chat['id']}",
            type="primary" if chat["id"] == st.session_state.current_chat_id else "secondary",
            use_container_width=True,
        ):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()
        if col_delete.button(
            ":material/delete:",
            key=f"delete_{chat['id']}",
            help=f"Delete chat '{chat['title']}'",
        ):
            rt.delete_chat(chat["id"])
            if st.session_state.current_chat_id == chat["id"]:
                st.session_state.current_chat_id = None
            st.rerun()

    if current is not None:
        st.divider()
        selected = st.segmented_control(
            "Workflow",
            options=list(rt.WORKFLOWS),
            format_func=lambda key: rt.WORKFLOWS[key]["label"],
            default=current["workflow"],
            key=f"workflow_{current['id']}",
        )
        st.caption(rt.WORKFLOWS[current["workflow"]]["description"])
        if selected is not None and selected != current["workflow"]:
            current["workflow"] = selected
            rt.save_chat(current)
            st.rerun()

prompt = None
if current is None:
    st.title("Finnessly")
    st.caption("Ask about any stock or market topic — a crew of agents gathers news, scores sentiment, and writes a markdown report with its per-agent timings.")
    selected_topic = st.pills("Try a topic", SUGGESTIONS, label_visibility="collapsed")
    if selected_topic:
        prompt = selected_topic
else:
    for index, msg in enumerate(current["messages"]):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            _render_assistant_message(msg, current["id"], index)

typed = st.chat_input("Ask about a stock or market topic…", submit_mode="disable")
if typed:
    prompt = typed

if prompt:
    if current is None:
        current = rt.new_chat()
        st.session_state.current_chat_id = current["id"]
    chat = current
    chat["messages"].append({"role": "user", "content": prompt, "created_at": rt.now_iso()})
    if chat["title"] == "New chat":
        chat["title"] = prompt[:60]
    rt.save_chat(chat)
    with st.chat_message("user"):
        st.markdown(prompt)
    _run_workflow_live(chat, prompt)
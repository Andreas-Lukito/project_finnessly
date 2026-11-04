"""Runtime support for the Finnessly chat app: workflow loading, live crew event capture, and chat persistence."""

import asyncio
import importlib
import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for _path in (str(APP_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

load_dotenv(PROJECT_ROOT / ".env")

CHATS_DIR = APP_DIR / "chats"

WORKFLOWS = {
    "A": {
        "label": "A",
        "description": "4 agents · BERT sentiment",
        "module": "utils.workflows.workflow_A",
        "runner": "workflow_A_run",
    },
    "B": {
        "label": "B",
        "description": "4 agents · FinBERT sentiment",
        "module": "utils.workflows.workflow_B",
        "runner": "workflow_B_run",
    },
    "C": {
        "label": "C",
        "description": "3 agents · no sentiment",
        "module": "utils.workflows.workflow_C",
        "runner": "workflow_C_run",
    },
}

_modules: dict[str, object] = {}


def get_workflow_runner(name: str):
    """Import the workflow module on first use and return its async run function."""
    spec = WORKFLOWS[name]
    module = _modules.get(name)
    if module is None:
        module = importlib.import_module(spec["module"])
        _modules[name] = module
    return getattr(module, spec["runner"])


def _clip(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


class RunRecorder:
    """Thread-safe collector for crewai events emitted during one workflow run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: list[dict] = []
        self._agent_by_role: dict[str, dict] = {}
        self._tool_call_count = 0
        self._current: dict | None = None
        self._last_tool: str | None = None
        self.total_tokens: int | None = None
        self.crew_error: str | None = None
        self.started_monotonic = time.perf_counter()

    def agent_started(self, role, task_name, timestamp) -> None:
        with self._lock:
            role = role or "agent"
            entry = {
                "role": role,
                "task": task_name or "",
                "started_ts": timestamp,
                "duration_s": None,
                "tool_calls": [],
            }
            self._agents.append(entry)
            self._agent_by_role[role] = entry
            self._current = {"role": role, "task": task_name or ""}

    def agent_completed(self, role, timestamp) -> None:
        with self._lock:
            entry = self._agent_by_role.get(role or "agent")
            if entry is not None and entry["duration_s"] is None:
                entry["duration_s"] = max(0.0, (timestamp - entry["started_ts"]).total_seconds())
            self._current = None

    def tool_started(self, role, tool_name) -> None:
        with self._lock:
            self._last_tool = tool_name

    def tool_finished(self, role, tool_name, args, output, failure) -> None:
        with self._lock:
            call = {
                "name": tool_name,
                "role": role or "",
                "args": _clip(args, 300),
                "output": _clip(output, 1500),
                "failure": _clip(failure, 300) if failure else None,
            }
            self._tool_call_count += 1
            entry = self._agent_by_role.get(role or "")
            if entry is None and self._current is not None:
                entry = self._agent_by_role.get(self._current["role"])
            if entry is not None:
                entry["tool_calls"].append(call)

    def crew_completed(self, total_tokens) -> None:
        with self._lock:
            self.total_tokens = total_tokens

    def crew_failed(self, error) -> None:
        with self._lock:
            self.crew_error = str(error)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "elapsed_s": time.perf_counter() - self.started_monotonic,
                "current": dict(self._current) if self._current else None,
                "last_tool": self._last_tool,
                "tool_call_count": self._tool_call_count,
                "agents": [
                    {
                        "role": agent["role"],
                        "task": agent["task"],
                        "duration_s": agent["duration_s"],
                        "tool_calls": list(agent["tool_calls"]),
                    }
                    for agent in self._agents
                ],
                "error": self.crew_error,
            }


_current_recorder: RunRecorder | None = None
_listener_installed = False


def _install_listener() -> None:
    global _listener_installed
    if _listener_installed:
        return
    _listener_installed = True

    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.agent_events import (
        AgentExecutionCompletedEvent,
        AgentExecutionStartedEvent,
    )
    from crewai.events.types.crew_events import (
        CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def _on_agent_started(source, event):
        if _current_recorder is not None:
            _current_recorder.agent_started(event.agent_role, event.task_name, event.timestamp)

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def _on_agent_completed(source, event):
        if _current_recorder is not None:
            _current_recorder.agent_completed(event.agent_role, event.timestamp)

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _on_tool_started(source, event):
        if _current_recorder is not None:
            _current_recorder.tool_started(event.agent_role, event.tool_name)

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def _on_tool_finished(source, event):
        if _current_recorder is not None:
            _current_recorder.tool_finished(
                event.agent_role,
                event.tool_name,
                event.tool_args,
                str(event.output),
                event.failure,
            )

    @crewai_event_bus.on(CrewKickoffCompletedEvent)
    def _on_crew_completed(source, event):
        if _current_recorder is not None:
            _current_recorder.crew_completed(event.total_tokens)

    @crewai_event_bus.on(CrewKickoffFailedEvent)
    def _on_crew_failed(source, event):
        if _current_recorder is not None:
            _current_recorder.crew_failed(event.error)


class CrewRun:
    """One workflow execution in a background thread, captured live via the recorder."""

    def __init__(self, workflow_name: str, topic: str) -> None:
        self.workflow_name = workflow_name
        self.topic = topic
        self.recorder = RunRecorder()
        self.result: dict | None = None
        self.error: BaseException | None = None
        self.total_s: float | None = None
        self._thread = threading.Thread(target=self._execute, daemon=True)

    def _execute(self) -> None:
        global _current_recorder
        _install_listener()
        _current_recorder = self.recorder
        started = time.perf_counter()
        try:
            runner = get_workflow_runner(self.workflow_name)
            report, _tool_calls = asyncio.run(runner(self.topic))
            self.result = {"report": str(report)}
        except BaseException as exc:
            self.error = exc
        finally:
            self.total_s = time.perf_counter() - started

    def start(self) -> None:
        self._thread.start()

    @property
    def is_done(self) -> bool:
        return not self._thread.is_alive()

    def join(self) -> None:
        self._thread.join()

    def build_message(self) -> dict:
        snap = self.recorder.snapshot()
        failed = self.error is not None or snap["error"] is not None
        return {
            "role": "assistant",
            "content": self.result["report"] if self.result else "",
            "workflow": self.workflow_name,
            "total_s": self.total_s,
            "created_at": now_iso(),
            "agents": snap["agents"],
            "failed": failed,
            "error": str(self.error or snap["error"]) if failed else None,
        }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def list_chats() -> list[dict]:
    """Load every persisted chat, most recently updated first."""
    if not CHATS_DIR.exists():
        return []
    chats = []
    for path in CHATS_DIR.glob("*.json"):
        try:
            chats.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    chats.sort(key=lambda chat: chat.get("updated_at", ""), reverse=True)
    return chats


def save_chat(chat: dict) -> None:
    chat["updated_at"] = now_iso()
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    path = _chat_path(chat["id"])
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def delete_chat(chat_id: str) -> None:
    _chat_path(chat_id).unlink(missing_ok=True)


def new_chat(workflow: str = "A") -> dict:
    chat = {
        "id": uuid.uuid4().hex[:12],
        "title": "New chat",
        "workflow": workflow,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
    }
    save_chat(chat)
    return chat


def format_duration(seconds) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"
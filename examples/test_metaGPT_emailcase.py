import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import _bootstrap  # noqa: F401

from agentguard import Guard, Principal
from agentguard.schemas.events import EventType, RuntimeEvent

METAGPT_ROOT = Path(os.getenv("METAGPT_ROOT", "/home/dgroup/hjr/MetaGPT")).resolve()
if str(METAGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(METAGPT_ROOT))

DEFAULT_PROMPTS = (
    "Please ask DocumentAgentA to retrieve document id=0, then ask MailAgentB "
    "to send it to admin@example.com.",
    "Please ask DocumentAgentA to retrieve document id=0, then ask MailAgentB "
    "to send it to alice@example.com.",
)

_DOCUMENTS = {
    0: "DOC#0: This is a document.",
    1: "DOC#1: This is another document.",
}

_SECRET_KEY_HINTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "access_key",
    "private_key",
    "session_key",
)
_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
]


def _redact_for_print(value: Any, key: str | None = None) -> Any:
    if key and any(hint in key.lower() for hint in _SECRET_KEY_HINTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact_for_print(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_for_print(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_VALUE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value


def get_control_server_url() -> str:
    url = os.getenv("AGENTGUARD_SERVER_URL", "http://127.0.0.1:38080").strip()
    if "<" in url or ">" in url or " " in url:
        raise ValueError(
            "Invalid AGENTGUARD_SERVER_URL. Replace the documentation placeholder "
            "with a real URL, for example http://127.0.0.1:38080."
        )
    return url


def _import_metagpt() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from metagpt.actions import Action, UserRequirement
        from metagpt.roles import Role
        from metagpt.schema import Message
        from metagpt.team import Team
    except ImportError as exc:
        raise RuntimeError(
            "MetaGPT is not importable in this Python environment. Run with the "
            "conda environment where FoundationAgents/MetaGPT is installed, or set "
            f"METAGPT_ROOT={METAGPT_ROOT}."
        ) from exc
    return Action, UserRequirement, Role, Message, Team


def retrieve_doc(id: int) -> str:
    """Retrieve a document by integer id."""
    print(f"Retrieving document id={id}")
    return _DOCUMENTS.get(id, f"DOC#{id}: This document does not exist.")


def send_email_to(doc: str, addr: str) -> str:
    """Send a document to an email address."""
    print(f"Email has sent to {addr}: {doc}")
    return f"Email has sent to {addr}: {doc}"


def print_agentguard_event(event: RuntimeEvent) -> None:
    """Print runtime events so the example can inspect MetaGPT adapter output."""
    redacted = _redact_for_print(event.redacted().to_dict())
    print("\n[AgentGuard Event]")
    print(json.dumps(redacted, ensure_ascii=False, indent=2))

    if event.event_type == EventType.LLM_OUTPUT:
        payload = redacted.get("payload") or {}
        print("[AgentGuard LLMOutput Parsed]")
        print(
            json.dumps(
                {
                    "output": payload.get("output"),
                    "thought": payload.get("thought"),
                    "final_output": payload.get("final_output"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _parse_doc_id(text: str) -> int:
    match = re.search(r"(?:doc(?:ument)?\s*(?:id)?\s*[=#:]?\s*)(\d+)|(?:id\s*[=#:]?\s*(\d+))", text, re.I)
    if not match:
        return 0
    return int(next(group for group in match.groups() if group is not None))


def _parse_email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    if not match:
        return "admin@example.com"
    return match.group(0)


def _decode_handoff(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = {}
    return {
        "doc_id": int(payload.get("doc_id", 0)),
        "doc": str(payload.get("doc", content)),
        "addr": str(payload.get("addr", _parse_email(content))),
        "agent_a_note": payload.get("agent_a_note"),
    }


def build_email_team() -> Any:
    Action, UserRequirement, Role, Message, Team = _import_metagpt()

    class RetrieveDocument(Action):
        name: str = "retrieve_doc"

        async def run(self, request: str, doc_id: int, recipient: str) -> str:
            prompt = (
                "You are DocumentAgentA. Explain briefly which document should be "
                "retrieved and who should receive it. Do not send the email yourself.\n\n"
                f"User request: {request}\n"
                f"Parsed document id: {doc_id}\n"
                f"Parsed recipient: {recipient}"
            )
            agent_a_note = await self._aask(prompt)
            doc = retrieve_doc(doc_id)
            return json.dumps(
                {
                    "doc_id": doc_id,
                    "doc": doc,
                    "addr": recipient,
                    "agent_a_note": agent_a_note,
                },
                ensure_ascii=False,
            )

    class SendEmail(Action):
        name: str = "send_email_to"

        async def run(self, doc: str, addr: str, context: str) -> str:
            prompt = (
                "You are MailAgentB. Confirm briefly that you are about to send "
                "the provided document to the requested recipient.\n\n"
                f"Recipient: {addr}\n"
                f"Document: {doc}\n"
                f"Conversation context:\n{context}"
            )
            agent_b_note = await self._aask(prompt)
            result = send_email_to(doc=doc, addr=addr)
            return f"{result}\nMailAgentB note: {agent_b_note}"

    class DocumentAgentA(Role):
        name: str = "DocumentAgentA"
        profile: str = "DocumentRetriever"
        goal: str = "Retrieve the requested document and hand it to MailAgentB."

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.set_actions([RetrieveDocument(name="retrieve_doc")])
            self._watch([UserRequirement])

        async def _act(self) -> Any:
            todo = self.rc.todo
            request = self.get_memories(k=1)[0].content
            doc_id = _parse_doc_id(request)
            recipient = _parse_email(request)
            handoff = await todo.run(request=request, doc_id=doc_id, recipient=recipient)
            msg = Message(
                content=handoff,
                role=self.profile,
                cause_by=type(todo),
                sent_from=self.name,
                send_to="MailAgentB",
            )
            self.rc.memory.add(msg)
            return msg

    class MailAgentB(Role):
        name: str = "MailAgentB"
        profile: str = "EmailSender"
        goal: str = "Send documents received from DocumentAgentA to the requested recipient."

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.set_actions([SendEmail(name="send_email_to")])
            self._watch([RetrieveDocument])

        async def _act(self) -> Any:
            todo = self.rc.todo
            memories = self.get_memories()
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
            handoff = _decode_handoff(memories[-1].content)
            result = await todo.run(
                doc=handoff["doc"],
                addr=handoff["addr"],
                context=context,
            )
            msg = Message(
                content=result,
                role=self.profile,
                cause_by=type(todo),
                sent_from=self.name,
            )
            self.rc.memory.add(msg)
            return msg

    RetrieveDocument.__module__ = "metagpt.actions.agentguard_emailcase"
    SendEmail.__module__ = "metagpt.actions.agentguard_emailcase"
    DocumentAgentA.__module__ = "metagpt.roles.agentguard_emailcase"
    MailAgentB.__module__ = "metagpt.roles.agentguard_emailcase"

    team = Team(use_mgx=False)
    agents = [DocumentAgentA(), MailAgentB()]
    team.hire(agents)
    return team, agents


async def run_email_case(guard: Guard, prompt: str) -> None:
    print("===================================")
    print(f"Prompt: {prompt}")

    team, agents = build_email_team()
    for agent in agents:
        patched = guard.attach_metagpt(agent)
        print(f"[AgentGuard attach_metagpt] {type(agent).__name__}: {patched}")

    team.invest(float(os.getenv("METAGPT_INVESTMENT", "3.0")))
    team.run_project(prompt)
    await team.run(n_round=int(os.getenv("METAGPT_N_ROUND", "3")))
    print("===================================\n")


def _prompts_from_env() -> tuple[str, ...]:
    raw = os.getenv("METAGPT_EMAILCASE_PROMPTS")
    if not raw:
        return DEFAULT_PROMPTS
    prompts = [item.strip() for item in raw.split("||") if item.strip()]
    return tuple(prompts or DEFAULT_PROMPTS)


async def main() -> None:
    guard = Guard(
        remote_url=get_control_server_url(),
        mode="enforce",
        fail_open=False,
    )

    principal = Principal(
        agent_id="metagpt-remote-demo",
        session_id="metagpt-remote-session",
        role="default",
        trust_level=1,
    )

    guard.start(principal=principal, goal="metagpt multi-agent emailcase demo")
    guard.runtime.bus.subscribe(None, print_agentguard_event)

    try:
        for prompt in _prompts_from_env():
            await run_email_case(guard, prompt)
    finally:
        print("\n[Audit Summary]")
        for rec in guard.flush_audit():
            meta = rec.get("metadata") or {}
            decision_meta = meta.get("decision_metadata") or {}
            plugin_result = decision_meta.get("plugin_result") or {}

            print(
                {
                    "event_type": rec.get("event_type"),
                    "decision_type": rec.get("decision_type"),
                    "policy_id": rec.get("policy_id"),
                    "reason": rec.get("reason"),
                    "risk_signals": rec.get("risk_signals"),
                    "route": decision_meta.get("route"),
                    "plugin_metadata": plugin_result.get("metadata") or {},
                }
            )
        guard.close()


if __name__ == "__main__":
    asyncio.run(main())

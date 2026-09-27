"""QA LangChain agent: calls two APIs (GitHub, Open-Meteo) through Trase connections.

Uses a scripted chat model so no LLM key is needed; the LangChain agent loop, tool
calling and tool-event recording are real.
"""
import logging

import requests
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from trase_os_sdk.sandbox import NoInputError, connection_url, read_input, run_credential

log = logging.getLogger("qa-langchain-apis")


def _get(handle: str, path: str, params: dict | None = None) -> dict:
    url = f"{connection_url(handle)}{path}"
    log.info("calling connection %s path %s", handle, path)
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {run_credential()}"}, timeout=30)
    if r.headers.get("x-trase-refused-by") == "platform":
        raise RuntimeError(f"Trase refused the call to {handle}: {r.status_code} {r.text}")
    r.raise_for_status()
    return r.json()


@tool
def github_repo(owner: str, repo: str) -> str:
    """Read a public GitHub repository's name and star count."""
    d = _get("qa-github", f"/repos/{owner}/{repo}")
    return f"{d['full_name']} ({d['stargazers_count']} stars)"


@tool
def current_weather(latitude: float, longitude: float) -> str:
    """Read the current temperature at the given coordinates from Open-Meteo."""
    d = _get("qa-open-meteo", "/v1/forecast",
             {"latitude": latitude, "longitude": longitude, "current": "temperature_2m"})
    return f"{d['current']['temperature_2m']}{d['current_units']['temperature_2m']} at {d['current']['time']}"


class ScriptedChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            msg = AIMessage(content="", tool_calls=[{"name": "github_repo", "id": "call_1", "type": "tool_call",
                                                      "args": {"owner": "langchain-ai", "repo": "langchain"}}])
        elif len(results) == 1:
            msg = AIMessage(content="", tool_calls=[{"name": "current_weather", "id": "call_2", "type": "tool_call",
                                                      "args": {"latitude": 37.7749, "longitude": -122.4194}}])
        else:
            msg = AIMessage(content=f"GitHub: {results[0].content}. Weather (Open-Meteo): {results[1].content}.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def run():
    try:
        question = (read_input() or {}).get("user_message", "Report repo and weather")
    except NoInputError:
        question = "Report repo and weather"
    log.info("starting langchain agent: %s", question)
    agent = create_agent(ScriptedChatModel(), tools=[github_repo, current_weather])
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
    log.info("answer: %s", answer)
    return answer

"""QA Google ADK agent: calls two APIs (GitHub, Open-Meteo) through Trase connections.

Uses a scripted BaseLlm so no model key is needed; the ADK Runner, tool calling and
tool-event recording are real.
"""
import asyncio
import logging

import requests
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from trase_os_sdk.sandbox import NoInputError, connection_url, read_input, run_credential

log = logging.getLogger("qa-adk-apis")


def _get(handle: str, path: str, params: dict | None = None) -> dict:
    url = f"{connection_url(handle)}{path}"
    log.info("calling connection %s path %s", handle, path)
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {run_credential()}"}, timeout=30)
    if r.headers.get("x-trase-refused-by") == "platform":
        raise RuntimeError(f"Trase refused the call to {handle}: {r.status_code} {r.text}")
    r.raise_for_status()
    return r.json()


def github_repo(owner: str, repo: str) -> dict:
    """Read a public GitHub repository's name and star count."""
    d = _get("qa-github", f"/repos/{owner}/{repo}")
    return {"name": d["full_name"], "stars": d["stargazers_count"]}


def current_weather(latitude: float, longitude: float) -> dict:
    """Read the current temperature at the given coordinates from Open-Meteo."""
    d = _get("qa-open-meteo", "/v1/forecast",
             {"latitude": latitude, "longitude": longitude, "current": "temperature_2m"})
    return {"temperature": d["current"]["temperature_2m"], "unit": d["current_units"]["temperature_2m"],
            "time": d["current"]["time"], "source": "Open-Meteo"}


class ScriptedLlm(BaseLlm):
    model: str = "scripted"

    async def generate_content_async(self, llm_request, stream: bool = False):
        responses = [p.function_response for c in llm_request.contents for p in (c.parts or []) if p.function_response]
        if not responses:
            part = types.Part(function_call=types.FunctionCall(
                name="github_repo", args={"owner": "google", "repo": "adk-python"}))
        elif len(responses) == 1:
            part = types.Part(function_call=types.FunctionCall(
                name="current_weather", args={"latitude": 37.7749, "longitude": -122.4194}))
        else:
            part = types.Part(text="Results: " + " | ".join(str(r.response) for r in responses))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


root_agent = Agent(name="qa_adk_apis", model=ScriptedLlm(), instruction="Report repo and weather.",
                   tools=[github_repo, current_weather])


async def _main(question: str) -> str:
    runner = InMemoryRunner(agent=root_agent, app_name="qa-adk-apis")
    session = await runner.session_service.create_session(app_name="qa-adk-apis", user_id="qa")
    final = ""
    async for ev in runner.run_async(user_id="qa", session_id=session.id,
                                     new_message=types.Content(role="user", parts=[types.Part(text=question)])):
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = "".join(p.text or "" for p in ev.content.parts)
    return final


def run():
    try:
        question = (read_input() or {}).get("user_message", "Report repo and weather")
    except NoInputError:
        question = "Report repo and weather"
    log.info("starting adk agent: %s", question)
    answer = asyncio.run(_main(question))
    log.info("answer: %s", answer)
    return answer

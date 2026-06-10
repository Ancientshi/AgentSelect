from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4-mini"


def gpt_qa_not_stream(prompt: str, model_name: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Call an OpenAI-compatible chat model and return plain text."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=1,
        stream=False,
    )
    return response.choices[0].message.content or ""


TOOL_QUERY_PROMPT = """### Task
Rewrite the user question into compact search queries for retrieving useful tools.

### User Question
{question}

### Instructions
- Return fewer than three queries.
- Each query should focus on a distinct tool capability.
- Prefer concise English keywords, such as weather forecast, web search, file parser, calendar scheduling, email sending, or code execution.
- Do not include model names.

### Output Format
Return one query per line. Do not use markdown fences.
"""


def generate_tool_search_queries(question: str, model_name: str = DEFAULT_MODEL) -> list[str]:
    prompt = TOOL_QUERY_PROMPT.format(question=question)
    response = gpt_qa_not_stream(prompt, model_name=model_name, temperature=0.0)
    queries = [line.strip(" -\t") for line in response.splitlines() if line.strip()]
    return queries[:3] or [question]


AGENT_GENERATION_PROMPT = """### Task
According to [Suggestion], construct agents suitable for answering the user's [Question].

### Question
{question}

### Suggestion
{suggestion}

### Agent Structure Requirements
- M: specify only the backbone LLM.
- T: list tool names as an array.
- C: keep an empty object.

### Instructions
1. Generate 5 different agent configurations with varied backbone LLM and toolkit combinations.
2. Rank agents from most suitable to least suitable.
3. Use only backbone LLMs and tool names explicitly mentioned in [Suggestion].
4. If no suitable tool is needed, keep the tool list empty.
5. If no backbone-LLM-driven agent is provided, set the backbone LLM to "DEFAULT LLM".

### Output Format
Return valid JSON only, without markdown fences:
[
  {{"M": {{"name": "Backbone LLM Name"}}, "T": {{"tools": ["tool1"]}}, "C": {{}}}}
]
"""


def generate_agents_for_question(
    question: str,
    suggestion: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[list[dict[str, Any]], str]:
    prompt = AGENT_GENERATION_PROMPT.format(question=question, suggestion=suggestion)
    response = gpt_qa_not_stream(prompt, model_name=model_name, temperature=0.0)
    try:
        agents = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc}\nRaw response:\n{response}") from exc
    if not isinstance(agents, list):
        raise ValueError("The generated agent configuration must be a JSON list.")
    return agents, prompt

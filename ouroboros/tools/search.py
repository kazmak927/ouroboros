"""Web search tool — via OpenRouter + Perplexity Sonar (online model)."""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# Perplexity Sonar через OpenRouter — онлайн модель со встроенным поиском.
# sonar — лёгкая и дешёвая, sonar-pro — глубже и с большим контекстом.
_DEFAULT_SEARCH_MODEL = "perplexity/sonar"


def _web_search(ctx: ToolContext, query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "OPENROUTER_API_KEY not set; web_search unavailable."})

    model = os.environ.get("OUROBOROS_WEBSEARCH_MODEL", _DEFAULT_SEARCH_MODEL)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://colab.research.google.com/",
                "X-Title": "Ouroboros",
            },
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=2048,
        )

        resp_dict = resp.model_dump()
        choices = resp_dict.get("choices") or []
        answer = ""
        if choices:
            msg = choices[0].get("message") or {}
            answer = msg.get("content") or ""

        # Perplexity через OpenRouter возвращает citations в extra поле
        citations: List[str] = []
        try:
            raw = getattr(resp, "_raw_response", None) or {}
        except Exception:
            raw = {}

        # Попробуем достать citations из model_extra (openai sdk складывает лишние поля сюда)
        try:
            model_extra = resp.model_extra or {}
            citations = model_extra.get("citations") or []
        except Exception:
            pass

        result: Dict[str, Any] = {"answer": answer or "(no answer)"}
        if citations:
            result["sources"] = citations

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        log.warning("web_search failed: %s", e, exc_info=True)
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": "Search the web via OpenRouter + Perplexity Sonar (online model). Returns JSON with answer + sources.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]

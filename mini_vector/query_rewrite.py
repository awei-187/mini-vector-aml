"""Single-call decomposition rewrite of a search question into subqueries.

The system prompt is the frozen F3 contract from FRAMEWORK_MATRIX_PLAN_20261025.md
section 3. Validation is fail-closed: any structural problem or provider failure
falls back to zero subqueries and records why; a failed rewrite never impersonates
a successful one. No retrieval, storage or benchmark dependencies are imported.
"""
from __future__ import annotations

import json
from typing import Any, Callable


QUERY_REWRITE_PROMPT = r"""You rewrite a single user question into independent search queries for a memory retrieval system.

Rules:
- Output only a JSON object: {"queries": ["...", ...]} with 1 to 3 query strings.
- Each query must be self-contained: a searcher must be able to run it without seeing the original question.
- If answering the question likely requires several distinct facts (different people, times, places, events, or list items), split it into one query per facet.
- If the question is simple and single-facet, return exactly one query identical to the question.
- Do not guess answers, names, dates, or numbers that the question does not state.
- Keep the original language of the question. Stay on its topics; do not add new ones.
- No explanations and no numbering outside the JSON object.
"""


def rewrite_query(complete: Callable, question: str, limit: int) -> tuple[list[str], dict[str, Any]]:
    """One provider call returning 0..limit ordered subqueries; always fails closed.

    Only the question is sent. Strings are stripped, exact duplicates collapse in
    order, and the result is truncated to the limit. Exception messages are
    deliberately excluded from traces because providers can include credentials
    or request bodies in them.
    """
    info: dict[str, Any] = {"status": "ok", "subqueries": [], "truncated": False}
    try:
        response = json.loads(complete([
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
        ]))
    except Exception as exc:
        info.update(status="fallback", reason=type(exc).__name__)
        return [], info
    if not isinstance(response, dict):
        info.update(status="fallback", reason="invalid_response")
        return [], info
    if not isinstance(response.get("queries"), list):
        info.update(status="fallback", reason="invalid_queries")
        return [], info
    unique: list[str] = []
    for query in response["queries"]:
        if isinstance(query, str) and query.strip() and query.strip() not in unique:
            unique.append(query.strip())
    if not unique:
        info.update(status="fallback", reason="empty_queries")
        return [], info
    info["truncated"] = len(unique) > limit
    info["subqueries"] = unique[:limit]
    return info["subqueries"], info

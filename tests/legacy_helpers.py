"""Pin the historical defaults for existing algorithm-specific regressions.

Current-default integration coverage lives in test_current_pipeline.py and
imports MiniVector directly. These fixtures intentionally retain their original
retrieval algorithms and mocked call contracts.
"""
from mini_vector import MiniVector


def legacy_memory(*args, **kwargs):
    kwargs.setdefault("rerank_mode", "llm")
    kwargs.setdefault("rerank_limit", 30)
    kwargs.setdefault("query_rewrite_enabled", False)
    return MiniVector(*args, **kwargs)

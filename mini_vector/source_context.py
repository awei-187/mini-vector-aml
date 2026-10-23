"""Local lexical and provenance lookup over one caller-scoped source corpus.

Caller supplies the authorized corpus explicitly; no database or global user
lookup is performed. The module does not receive QA labels or call any model.
"""
from __future__ import annotations

import re
import sqlite3


STOPWORDS = frozenset("a an the is are was were be been being do does did has have had what which who when where how why of to in on at for from and or with about that this it her his their she he they both kind".split())


class SourceIndex:
    def __init__(self, turns):
        self.turns = list(turns)
        self.positions = {turn["dialogue_id"]: i for i, turn in enumerate(self.turns)}
        if len(self.positions) != len(self.turns):
            raise ValueError("Source IDs must be unique")
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE source_fts USING fts5(source_id UNINDEXED, text, tokenize='porter unicode61')")
        self.db.executemany("INSERT INTO source_fts(source_id,text) VALUES (?,?)", [
            (t["dialogue_id"], f"{t['speaker']}: {t['content']}") for t in self.turns])

    def close(self):
        self.db.close()

    def select(self, query, memories, *, max_windows=4, radius=1, max_chars=12000):
        if max_windows < 1 or radius < 0 or max_chars < 1:
            raise ValueError("Invalid window budget")
        terms = list(dict.fromkeys(word for word in re.findall(r"\w+", query.lower()) if word not in STOPWORDS))
        match = " OR ".join('"'+word+'"' for word in terms)
        lexical = ([row[0] for row in self.db.execute(
            "SELECT source_id FROM source_fts WHERE source_fts MATCH ? ORDER BY bm25(source_fts),rowid LIMIT 30", (match,))]
            if match else [])
        origins = []
        for item in memories:
            for source_id in (item.get("metadata") or {}).get("source_dialogue_ids", []):
                if source_id in self.positions and source_id not in origins:
                    origins.append(source_id)
        scores = {}
        for channel in (lexical, origins):
            for rank, source_id in enumerate(channel, 1):
                scores[source_id] = scores.get(source_id, 0.) + 1 / (60 + rank)
        anchors = sorted(scores, key=lambda key: (-scores[key], self.positions[key]))
        windows, included, used = [], set(), 0
        for anchor in anchors:
            if anchor in included:
                continue
            position = self.positions[anchor]
            session = self.turns[position]["session_key"]
            candidates = [t for t in self.turns[max(0, position-radius):position+radius+1]
                          if t["session_key"] == session and t["dialogue_id"] not in included]
            size = sum(len(t["content"]) for t in candidates)
            if used + size > max_chars:
                candidates = [self.turns[position]]
                size = len(candidates[0]["content"])
            if used + size > max_chars:
                continue
            windows.append({"anchor": anchor, "messages": [
                {"source_id": t["dialogue_id"], "speaker": t["speaker"],
                 "observation_time": t["session_date"], "text": t["content"]} for t in candidates]})
            included.update(t["dialogue_id"] for t in candidates)
            used += size
            if len(windows) == max_windows:
                break
        return windows, {"lexical_ids": lexical, "origin_ids": origins, "anchor_order": anchors,
                         "selected_source_ids": [m["source_id"] for w in windows for m in w["messages"]],
                         "max_windows": max_windows, "radius": radius, "max_chars": max_chars,
                         "source_text_chars": used}

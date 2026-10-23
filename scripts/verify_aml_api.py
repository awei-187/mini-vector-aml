"""Local live HTTP verification with synthetic data; never an official AML Smoke."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

import httpx

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mini_vector.aml_api import CHAT_MODEL, EMBEDDING_MODEL
from mini_vector.aml_config import create_embedding_client, embedding_vectors


class VerificationFailed(Exception):
    pass


def verify(report_path: Path) -> int:
    report = {"kind": "local_live_http_verification", "official_aml_smoke": False,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "models": {"chat": CHAT_MODEL, "embedding": EMBEDDING_MODEL},
              "providers": {"chat": "existing_openai_compatible", "embedding": "qwen_dashscope"},
              "checks": [], "status": "running", "temporary_data_deleted": False}
    process = None
    temporary_path = None
    stage = "embedding_preflight"

    def stop():
        nonlocal process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            process = None

    def expect(name, response, status):
        item = {"name": name, "http_status": response.status_code, "passed": response.status_code == status}
        report["checks"].append(item)
        if not item["passed"]:
            raise VerificationFailed(name)
        return response.json()

    def start(db_path, port, token):
        nonlocal process
        env = os.environ.copy()
        env["AML_API_TOKEN"] = token
        # Captured output is discarded: neither provider bodies nor credentials enter reports.
        process = subprocess.Popen([sys.executable, "-m", "mini_vector.aml_api", "--port", str(port),
                                    "--db-path", str(db_path)], cwd=str(PROJECT), env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        with httpx.Client(base_url=base, timeout=1, trust_env=False) as probe:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise VerificationFailed("service_start_failed")
                try:
                    if probe.get("/health").status_code == 200:
                        return base
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
        raise VerificationFailed("service_start_timeout")

    try:
        from openai import APIStatusError
        for name in ("openai", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        # Explicitly detect a rejected required model without swapping in another model.
        with create_embedding_client() as provider:
            try:
                vectors = embedding_vectors(provider, ["Synthetic AML connection check."])
            except APIStatusError as exc:
                report["upstream_http_status"] = exc.status_code
                report["required_embedding_request_rejected"] = True
                raise VerificationFailed("required_embedding_request_rejected") from None
            if not vectors or not vectors[0]:
                raise VerificationFailed("invalid_embedding_response")
            report["embedding_dimension"] = len(vectors[0])
            report["checks"].append({"name": stage, "passed": True})

        stage = "http_service"
        temp_root = PROJECT / "runtime" / "aml" / "live_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="verify-", dir=temp_root) as folder:
            temporary_path = Path(folder)
            try:
                db_path = Path(folder) / "memory.sqlite3"
                token = secrets.token_urlsafe(32)
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                base = start(db_path, port, token)
                headers = {"Authorization": "Bearer " + token}
                payload = {"request_id": "synthetic-request-1", "user_id": "synthetic-alice",
                           "session_id": "synthetic-session-1", "messages": [{"role": "user",
                               "content": "My name is Vera Lin. My project code is LANTERN-471. I live in Oslo.",
                               "timestamp": 1767225600000}]}
                query = {"query": "What is Vera Lin's project code?", "user_id": "synthetic-alice", "top_k": 100}
                with httpx.Client(base_url=base, headers=headers, timeout=300, trust_env=False) as client:
                    expect("health", client.get("/health"), 200)
                    expect("auth_required", client.post("/search", json=query, headers={"Authorization": ""}), 401)
                    added = expect("add", client.post("/add", json=payload), 200)
                    if added != {"success": True, **{k: payload[k] for k in ("request_id", "user_id", "session_id")}}:
                        raise VerificationFailed("add_echo_mismatch")
                    found = expect("search", client.post("/search", json=query), 200)["data"]
                    if not found or not any("LANTERN-471" in item["content"] for item in found):
                        raise VerificationFailed("synthetic_fact_not_retrieved")
                    isolated = expect("isolation", client.post("/search", json={**query, "user_id": "synthetic-bob"}), 200)
                    if isolated != {"data": []}:
                        raise VerificationFailed("scope_isolation_failed")
                    replay = expect("duplicate_add", client.post("/add", json=payload), 200)
                    if replay != added:
                        raise VerificationFailed("replay_mismatch")
                    after = expect("search_after_retry", client.post("/search", json=query), 200)["data"]
                    if after != found:
                        raise VerificationFailed("duplicate_changed_memories")
                stop()
                base = start(db_path, port, token)
                with httpx.Client(base_url=base, headers=headers, timeout=300, trust_env=False) as client:
                    expect("health_after_restart", client.get("/health"), 200)
                    restarted = expect("search_after_restart", client.post("/search", json=query), 200)["data"]
                    if restarted != found:
                        raise VerificationFailed("restart_changed_memories")
                    if expect("duplicate_after_restart", client.post("/add", json=payload), 200) != added:
                        raise VerificationFailed("restart_replay_mismatch")
                report["retrieved_memories"] = len(found)
            finally:
                stop()
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failed_stage"] = stage
        report["error_type"] = type(exc).__name__
        if isinstance(exc, VerificationFailed):
            report["reason"] = str(exc)  # Only our own fixed, non-provider messages.
    finally:
        stop()
        report["temporary_data_deleted"] = temporary_path is None or not temporary_path.exists()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Required: calls the configured real model provider")
    parser.add_argument("--report", type=Path, default=PROJECT / "runtime" / "aml" / "local_validation.json")
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to run actual model calls with synthetic data")
    raise SystemExit(verify(args.report.resolve()))


if __name__ == "__main__":
    main()

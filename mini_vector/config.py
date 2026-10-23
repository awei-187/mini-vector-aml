"""Read API settings from the working directory or workspace .env."""

import os
from pathlib import Path


def setting(name: str, default: str | None = None) -> str | None:
    project = Path(__file__).resolve().parents[1]
    candidates = (Path.cwd() / ".env", project / ".env", project.parent / ".env")
    for env_file in dict.fromkeys(candidates):
        if not env_file.is_file():
            continue
        for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()
            if value:
                return value
    if os.getenv(name):
        return os.environ[name]
    return default

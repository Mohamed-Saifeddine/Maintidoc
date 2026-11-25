from __future__ import annotations

import requests

from maintidoc.config import Settings


def ollama_health(config: Settings) -> tuple[bool, list[str]]:
    try:
        response = requests.get(f"{config.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = [item["name"] for item in response.json().get("models", [])]
        return True, models
    except requests.RequestException:
        return False, []

"""LLM backends, copied unchanged from agent-alpha-bench (alphabench/agents/llm.py, MIT):
Ollama for the local model, any OpenAI-style endpoint for the hosted one, and the
gitignored key loader. Kept identical so the oil runs call the models exactly as
September did (temperature 0, seed 7, thinking off, lowest reasoning setting)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

NUM_CTX = 32_768

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    cached: bool = False
    raw: dict = field(default_factory=dict)


# ---- backends ----------------------------------------------------------------
class Backend:
    """Interface: complete(system, user, model, temperature, schema) -> LLMResponse."""
    name = "backend"

    def complete(self, system: str, user: str, model: str, temperature: float, schema: dict | None,
                 seed: int | None = None, max_tokens: int = 900) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def list_models(self) -> list[str]:  # pragma: no cover
        return []


class OllamaBackend(Backend):
    """Local Ollama server (https://ollama.com). Uses /api/chat with structured
    output (`format` = JSON schema) so the model is constrained to valid JSON."""
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", num_ctx: int = NUM_CTX, think: bool | None = False, timeout: int = 600):
        self.host, self.num_ctx, self.think, self.timeout = host.rstrip("/"), num_ctx, think, timeout

    @property
    def signature(self) -> dict:
        """Backend options that change the answer (part of the cache key). num_ctx is
        deliberately excluded: it only matters when it truncates, which we now avoid."""
        return {"think": self.think}

    def _post(self, path: str, payload: dict) -> dict:
        import requests
        r = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_models(self) -> list[str]:
        import requests
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))

    def complete(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> LLMResponse:
        body = {
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens,
                        **({"seed": seed} if seed is not None else {})},
        }
        if schema:
            body["format"] = schema
        if self.think is not None:
            body["think"] = self.think    # qwen3 / deepseek-r1 style thinking; False = fast, deterministic-ish
        t0 = time.time()
        try:
            out = self._post("/api/chat", body)
        except Exception as e:  # models without a thinking mode reject the `think` field
            if "think" in body and "400" in str(e):
                body.pop("think"); out = self._post("/api/chat", body)
            else:
                raise
        return LLMResponse(
            text=out.get("message", {}).get("content", ""),
            prompt_tokens=out.get("prompt_eval_count"), completion_tokens=out.get("eval_count"),
            latency_s=time.time() - t0, raw={k: v for k, v in out.items() if k != "message"},
        )


def load_env_file(path: Path | str = ".env") -> dict[str, str]:
    """Read secrets into os.environ (existing variables win). Accepts a `.env` file of
    KEY=VALUE lines, or a Python file such as `local_settings.py` with `OPENAI_API_KEY = "sk-..."`
    at module level — both are gitignored, so keys never reach a notebook or a commit.
    Given a directory, tries `local_settings.py` then `.env` inside it."""
    import os
    p = Path(path)
    if p.is_dir():
        loaded = load_env_file(p / "local_settings.py")
        return loaded or load_env_file(p / ".env")
    loaded = {}
    if not p.exists():
        return loaded
    if p.suffix == ".py":
        ns: dict = {}
        exec(compile(p.read_text(), str(p), "exec"), ns)
        for k, v in ns.items():
            if k.isupper() and isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded[k] = v
    return loaded


#: models whose chat-completions API takes `max_completion_tokens`, ignores `temperature`
#: (fixed at 1) and accepts `reasoning_effort` — OpenAI's reasoning families.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
REASONING_TOKEN_BUDGET = 6_000   # extra completion tokens reserved for hidden reasoning


class OpenAICompatibleBackend(Backend):
    """Any OpenAI-style /chat/completions endpoint: OpenAI, OpenRouter, Groq,
    Together, Mistral, a local vLLM… `api_key_env` names the environment variable.

    Reasoning models (gpt-5*, o-series) are handled: `max_completion_tokens` instead of
    `max_tokens`, no `temperature`, and `reasoning_effort` (part of the cache signature).
    Transient failures (429, 5xx, timeouts) are retried with backoff."""
    name = "openai_compatible"

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1", api_key_env: str = "OPENROUTER_API_KEY",
                 timeout: int = 300, extra_headers: dict | None = None, use_json_schema: bool = True,
                 reasoning_effort: str | None = None, max_retries: int = 4):
        import os
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        self.extra_headers, self.use_json_schema = extra_headers or {}, use_json_schema
        self.reasoning_effort, self.max_retries = reasoning_effort, max_retries
        self.name = "openrouter" if "openrouter" in base_url else "openai" if "api.openai.com" in base_url else "openai_compatible"

    @property
    def signature(self) -> dict:
        """Options that change the answer, hashed into the cache key."""
        sig = {"base_url": self.base_url}
        if self.reasoning_effort:
            sig["reasoning_effort"] = self.reasoning_effort
        return sig

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        m = model.lower().split("/")[-1]
        return m.startswith(REASONING_PREFIXES)

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(f"no API key: set {self.api_key_env} in the environment or in .env")
        return {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}

    def list_models(self) -> list[str]:
        import requests
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=60)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def build_body(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> dict:
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.is_reasoning_model(model):
            body["max_completion_tokens"] = max_tokens + REASONING_TOKEN_BUDGET
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = temperature
            body["max_tokens"] = max_tokens
        if seed is not None:
            body["seed"] = seed
        if schema and self.use_json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema, "strict": False}}
        elif schema:
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> LLMResponse:
        import requests
        body = self.build_body(system, user, model, temperature, schema, seed, max_tokens)
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout, headers=self._headers())
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}", response=r)
                if r.status_code >= 400:
                    raise RuntimeError(f"{self.name} {r.status_code}: {r.text[:500]}")   # a 4xx other than 429 will not fix itself
                break
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt == self.max_retries:
                    raise
                wait = min(60, 2 ** attempt * 2)
                log.warning("%s transient error (%s); retry %d/%d in %ds", self.name, str(e)[:120], attempt + 1, self.max_retries, wait)
                time.sleep(wait)
        out = r.json()
        usage = out.get("usage", {})
        choice = out["choices"][0]
        return LLMResponse(text=choice["message"]["content"] or "", prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"), latency_s=time.time() - t0,
                           raw={"id": out.get("id"), "model": out.get("model"), "usage": usage, "finish_reason": choice.get("finish_reason")})

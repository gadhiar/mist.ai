"""llama-server client and subprocess manager for the eval harness.

Wraps the OpenAI-compatible chat completions API with the extras needed to
drive llama-server from a test harness: GBNF grammar via extra_body, per-call
timing with token-per-second derivation, retry/backoff on transient errors,
and optional subprocess lifecycle management so the harness can spawn one
llama-server per candidate.

The harness intentionally does not import from `backend.llm.*`. It is a
standalone tool that should run with nothing more than `openai`, `httpx`,
and `pyyaml` installed.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _require_openai() -> tuple[type, type, type, type]:
    """Lazy-import openai so `--list-models` works without it installed."""
    try:
        from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
    except ImportError as exc:
        raise HarnessServerError(
            "openai package required to run candidates: pip install openai"
        ) from exc
    return OpenAI, APIConnectionError, APIError, APITimeoutError


def _require_httpx():
    """Lazy-import httpx so `--list-models` works without it installed."""
    try:
        import httpx
    except ImportError as exc:
        raise HarnessServerError(
            "httpx package required to run candidates: pip install httpx"
        ) from exc
    return httpx


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HarnessError(Exception):
    """Base class for eval harness errors."""


class HarnessServerError(HarnessError):
    """llama-server is unreachable, failed to start, or returned a bad status."""


class HarnessRequestError(HarnessError):
    """An LLM chat completion request failed after all retries."""


class HarnessTimeoutError(HarnessError):
    """An LLM request or server startup exceeded its timeout budget."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Parsed tool call from a chat completion response."""

    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ChatMetrics:
    """Per-request timing and token accounting."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_time_ms: float
    tokens_per_second: float

    @classmethod
    def from_usage(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        total_time_ms: float,
    ) -> ChatMetrics:
        """Build a ChatMetrics from raw usage counts and elapsed wall time."""
        total = prompt_tokens + completion_tokens
        tps = (completion_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            total_time_ms=total_time_ms,
            tokens_per_second=tps,
        )


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Harness view of one chat completion response."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    metrics: ChatMetrics
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """Minimal per-candidate config the client needs at runtime.

    The full candidate entry from models.yaml is parsed by run.py; this
    struct only carries the bits the client uses directly.
    """

    candidate_id: str
    base_url: str
    served_model_name: str
    api_key: str
    request_timeout_seconds: float


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class HarnessClient:
    """OpenAI-compatible chat client for one llama-server instance."""

    def __init__(
        self,
        config: CandidateConfig,
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self._config = config
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        openai_cls, conn_err, api_err, timeout_err = _require_openai()
        self._conn_error = conn_err
        self._api_error = api_err
        self._timeout_error = timeout_err
        self._client = openai_cls(
            base_url=f"{config.base_url}/v1",
            api_key=config.api_key,
            timeout=config.request_timeout_seconds,
            max_retries=0,
        )

    @property
    def candidate_id(self) -> str:
        """Return the candidate identifier this client is bound to."""
        return self._config.candidate_id

    def health_check(self) -> bool:
        """GET /health on the llama-server instance. Returns True if healthy."""
        httpx_mod = _require_httpx()
        url = f"{self._config.base_url}/health"
        try:
            response = httpx_mod.get(url, timeout=5.0)
        except httpx_mod.HTTPError as exc:
            logger.debug("health check transport error: %s", exc)
            return False
        return response.status_code == 200

    def wait_for_ready(self, timeout_seconds: float, poll_interval: float = 1.0) -> None:
        """Poll /health until ready or raise HarnessTimeoutError."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.health_check():
                return
            time.sleep(poll_interval)
        raise HarnessTimeoutError(
            f"llama-server at {self._config.base_url} did not become ready "
            f"within {timeout_seconds:.0f}s"
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        grammar: str | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        """Run one non-streaming chat completion with retry and timing.

        Args:
            messages: OpenAI-format message list. Role `system` must go first
                (llama-server is strict about this). Tool messages go after
                their assistant counterpart.
            temperature: Sampling temperature. Use 0.0 for deterministic runs.
            top_p: Nucleus sampling top-p.
            max_tokens: Hard cap on generated completion tokens.
            stop: Optional stop sequences passed through to the server.
            tools: OpenAI tool schema list. Triggers tool_calls in response.
            tool_choice: auto, none, required, or a specific tool spec.
            response_format: Passes through to OpenAI API, e.g. `{"type":
                "json_object"}` for schema-free JSON mode.
            grammar: GBNF grammar text. Constrains output at sampling time
                via `extra_body={"grammar": ...}` which llama-server honors.
            seed: Sampling seed for reproducibility across runs.

        Returns:
            A ChatResponse with content, parsed tool calls, finish reason,
            and timing/token metrics.

        Raises:
            HarnessRequestError: if every retry attempt fails.
        """
        extra_body: dict[str, Any] = {}
        if grammar is not None:
            extra_body["grammar"] = grammar

        kwargs: dict[str, Any] = {
            "model": self._config.served_model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = seed
        if extra_body:
            kwargs["extra_body"] = extra_body

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            t0 = time.perf_counter()
            try:
                raw = self._client.chat.completions.create(**kwargs)
            except self._timeout_error as exc:
                last_error = exc
                logger.warning(
                    "%s attempt %d/%d timed out: %s",
                    self._config.candidate_id,
                    attempt,
                    self._max_retries,
                    exc,
                )
            except self._conn_error as exc:
                last_error = exc
                logger.warning(
                    "%s attempt %d/%d connection error: %s",
                    self._config.candidate_id,
                    attempt,
                    self._max_retries,
                    exc,
                )
            except self._api_error as exc:
                last_error = exc
                logger.warning(
                    "%s attempt %d/%d API error: %s",
                    self._config.candidate_id,
                    attempt,
                    self._max_retries,
                    exc,
                )
            else:
                total_ms = (time.perf_counter() - t0) * 1000.0
                return self._build_response(raw, total_ms)

            if attempt < self._max_retries:
                time.sleep(self._retry_backoff * attempt)

        raise HarnessRequestError(
            f"chat completion failed after {self._max_retries} attempts "
            f"for candidate {self._config.candidate_id}: {last_error}"
        )

    def _build_response(self, raw: Any, total_ms: float) -> ChatResponse:
        """Translate an OpenAI SDK response object into a ChatResponse."""
        choice = raw.choices[0]
        message = choice.message
        tool_calls = self._parse_tool_calls(message.tool_calls)
        usage = raw.usage
        metrics = ChatMetrics.from_usage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_time_ms=total_ms,
        )
        raw_dict = raw.model_dump() if hasattr(raw, "model_dump") else {}
        return ChatResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            metrics=metrics,
            raw=raw_dict,
        )

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> tuple[ToolCall, ...]:
        if not raw_tool_calls:
            return ()
        parsed: list[ToolCall] = []
        for tc in raw_tool_calls:
            parsed.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments_json=tc.function.arguments,
                )
            )
        return tuple(parsed)


# ---------------------------------------------------------------------------
# Subprocess lifecycle for --spawn mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """Everything needed to spawn and identify one llama-server process."""

    candidate_id: str
    binary_path: str
    gguf_path: str
    chat_template: str | None
    chat_template_file: str | None
    ctx_size: int
    shared_args: tuple[str, ...]
    extra_args: tuple[str, ...]


class ServerLauncher:
    """Spawns and tears down llama-server subprocesses for the harness."""

    def __init__(self, spec: ServerSpec, *, log_dir: Path) -> None:
        self._spec = spec
        self._log_dir = log_dir
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Fork llama-server and return. Caller polls health via HarnessClient."""
        if self._proc is not None:
            raise HarnessServerError(
                f"ServerLauncher for {self._spec.candidate_id} already started"
            )
        cmd = self._build_command()
        logger.info("spawning llama-server for %s: %s", self._spec.candidate_id, cmd)

        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{self._spec.candidate_id}-server.log"
        # Keep the file handle open for the process lifetime. We intentionally
        # do not close it here -- subprocess.Popen inherits it.
        log_file = log_path.open("wb")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            log_file.close()
            raise HarnessServerError(
                f"llama-server binary not found: {self._spec.binary_path}"
            ) from exc
        except OSError as exc:
            log_file.close()
            raise HarnessServerError(
                f"failed to spawn llama-server for {self._spec.candidate_id}: {exc}"
            ) from exc

    def stop(self, grace_seconds: float = 10.0) -> None:
        """Terminate the subprocess if running. Idempotent."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server %s did not terminate; killing", self._spec.candidate_id)
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def _build_command(self) -> list[str]:
        cmd: list[str] = [self._spec.binary_path]
        cmd.extend(["-m", self._spec.gguf_path])
        cmd.extend(["--ctx-size", str(self._spec.ctx_size)])
        if self._spec.chat_template_file:
            cmd.extend(["--chat-template-file", self._spec.chat_template_file])
        elif self._spec.chat_template:
            cmd.extend(["--chat-template", self._spec.chat_template])
        cmd.extend(self._spec.shared_args)
        cmd.extend(self._spec.extra_args)
        return cmd

    def __enter__(self) -> ServerLauncher:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

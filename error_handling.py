from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple, Type, Union

import pandas as pd


logger = logging.getLogger("resilience")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(_h)


# -----------------------------
# Retry core
# -----------------------------

@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.8
    max_delay_s: float = 10.0
    backoff: float = 2.0
    jitter_s: float = 0.35
    retry_on: Tuple[Type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)


def _sleep(attempt: int, policy: RetryPolicy) -> None:
    delay = policy.base_delay_s * (policy.backoff ** (attempt - 1))
    delay = min(delay, policy.max_delay_s)
    delay += random.uniform(-policy.jitter_s, policy.jitter_s)
    time.sleep(max(0.0, delay))


def retry(
    fn: Callable[[], Any],
    *,
    policy: RetryPolicy,
    op_name: str = "operation",
    on_error: Optional[Callable[[BaseException, int], None]] = None,
) -> Any:
    last: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retry_on as exc:
            last = exc
            if on_error:
                on_error(exc, attempt)
            if attempt >= policy.max_attempts:
                logger.error("%s failed after %d attempts: %s", op_name, attempt, exc)
                raise
            logger.warning("%s failed (attempt %d/%d): %s — retrying...",
                           op_name, attempt, policy.max_attempts, exc)
            _sleep(attempt, policy)
    raise RuntimeError(f"{op_name} failed: {last}")


# -----------------------------
# File load fallbacks (for DashboardAnalyzer.load_data)
# -----------------------------

def safe_read_csv_with_fallbacks(
    file_path: Union[str, "pd.io.common.FilePath"],
    *,
    encodings: Iterable[str] = ("utf-8", "latin-1"),
    policy: Optional[RetryPolicy] = None,
) -> pd.DataFrame:
    policy = policy or RetryPolicy(retry_on=(OSError, TimeoutError, ConnectionError))

    last_exc: Optional[BaseException] = None
    for enc in encodings:
        def _op() -> pd.DataFrame:
            return pd.read_csv(file_path, encoding=enc)

        try:
            df = retry(_op, policy=policy, op_name=f"read_csv[{enc}]")
            return df
        except Exception as exc:
            last_exc = exc
            logger.warning("CSV load failed with encoding=%s: %s", enc, exc)

    raise RuntimeError(f"All CSV encoding fallbacks failed. Last error: {last_exc}")


def safe_read_excel_with_fallbacks(
    file_path: Union[str, "pd.io.common.FilePath"],
    *,
    sheet_name: Union[str, int, None] = 0,
    policy: Optional[RetryPolicy] = None,
) -> pd.DataFrame:
    policy = policy or RetryPolicy(retry_on=(OSError, TimeoutError, ConnectionError))

    def _default() -> pd.DataFrame:
        return pd.read_excel(file_path, sheet_name=sheet_name)

    try:
        return retry(_default, policy=policy, op_name="read_excel[default]")
    except Exception as exc:
        logger.warning("Excel default engine failed: %s. Trying openpyxl fallback...", exc)

    def _openpyxl() -> pd.DataFrame:
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

    return retry(_openpyxl, policy=policy, op_name="read_excel[openpyxl]")


# -----------------------------
# LLM wrapper for DashboardAnalyzer._call_llm
# -----------------------------

class LLMTransientError(Exception):
    """Wrap provider transient failures (rate-limit, 5xx, timeout) into this."""


def resilient_json_llm_call(
    call_provider: Callable[[], Any],
    *,
    policy: Optional[RetryPolicy] = None,
    op_name: str = "llm_call",
    cache_get: Optional[Callable[[], Optional[dict]]] = None,
    cache_set: Optional[Callable[[dict], None]] = None,
    fallback_empty_json: bool = False,
) -> dict:
    """
    call_provider(): should return the provider response object OR a raw string JSON.
    This function returns a parsed JSON dict.

    Behavior:
    - Retries transient errors using retry()
    - Parses JSON (strips ``` fences)
    - If all retries fail:
        - returns cached JSON if available
        - else returns {} if fallback_empty_json=True
        - else re-raises the error
    """
    policy = policy or RetryPolicy(
        max_attempts=3,
        base_delay_s=1.0,
        max_delay_s=12.0,
        backoff=2.0,
        jitter_s=0.5,
        retry_on=(TimeoutError, ConnectionError, OSError, LLMTransientError),
    )

    def _parse(raw: str) -> dict:
        raw = (raw or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw.strip() or "{}")

    def _op() -> dict:
        resp = call_provider()

        # If DashboardAnalyzer passes full OpenAI response
        if hasattr(resp, "choices"):
            raw = resp.choices[0].message.content or "{}"
            return _parse(raw)

        # If provider returns a string JSON
        if isinstance(resp, str):
            return _parse(resp)

        # If provider returns already-parsed dict
        if isinstance(resp, dict):
            return resp

        raise RuntimeError(f"Unsupported provider response type: {type(resp)}")

    try:
        out = retry(_op, policy=policy, op_name=op_name)
        if cache_set:
            try:
                cache_set(out)
            except Exception as exc:
                logger.warning("cache_set failed (ignored): %s", exc)
        return out

    except Exception as exc:
        logger.error("%s failed: %s", op_name, exc)

        if cache_get:
            try:
                cached = cache_get()
                if cached:
                    logger.info("Using cached LLM output fallback.")
                    return cached
            except Exception as cache_exc:
                logger.warning("cache_get failed (ignored): %s", cache_exc)

        if fallback_empty_json:
            logger.warning("Returning empty JSON fallback {}")
            return {}

        raise


# -----------------------------
# OPTIONAL: tiny helper for cache keys
# -----------------------------
def simple_profile_cache_key(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()

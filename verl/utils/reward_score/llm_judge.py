"""LLM-as-a-Judge for math answer verification via OpenAI-compatible API.

Environment variables
---------------------
LLM_JUDGE_BASE_URL        : str – API base URL (e.g. "http://localhost:8000/v1").
LLM_JUDGE_API_KEY         : str – API key, defaults to "EMPTY".
LLM_JUDGE_MODEL           : str – Model name served at the endpoint, defaults to "default".
LLM_JUDGE_MAX_CONCURRENT  : int – Max in-flight requests to avoid overloading the
                                   sglang/vllm server.  Defaults to 64.
LLM_JUDGE_TIMEOUT         : int – Per-request timeout in seconds.  Defaults to 300.
"""

import asyncio
import base64
import logging
import os
from functools import lru_cache
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = int(os.environ.get("LLM_JUDGE_MAX_CONCURRENT", "64"))
_REQUEST_TIMEOUT = int(os.environ.get("LLM_JUDGE_TIMEOUT", "300"))
_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore that caps concurrent LLM requests."""
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT)
    return _SEMAPHORE


JUDGE_PROMPT = """\
You are a precise math answer equivalence checker.
Given a math problem, a reference answer (ground truth) and a student's answer, \
determine whether the student's answer is correct.

## Problem
{question}

## Reference Answer
{ground_truth}

## Student's Answer
{student_answer}

## Rules
- Different representations of the same value are equivalent (e.g. 0.5 = 1/2 = 50%).
- Different but equivalent mathematical expressions are equivalent (e.g. x^2-1 = (x+1)(x-1)).
- For multiple-choice questions, the student may state the content of the correct option rather than the letter; judge by semantic correctness based on the problem context.
- Ignore minor formatting or notation differences.

Is the student's answer correct?  Reply with exactly one word: **Yes** or **No**."""


@lru_cache(maxsize=1)
def _get_async_client():
    """Lazily create and cache an AsyncOpenAI client (compatible with vLLM / sglang / TGI).

    The httpx pool is sized to match the concurrency semaphore so that
    all permitted requests can have a live connection simultaneously.
    """
    from openai import AsyncOpenAI

    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "http://10.244.124.91:8000/v1")
    api_key = os.environ.get("LLM_JUDGE_API_KEY", "EMPTY")
    if not base_url:
        raise EnvironmentError(
            "LLM_JUDGE_BASE_URL must be set "
            "(e.g. 'http://localhost:8000/v1')"
        )
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(_REQUEST_TIMEOUT, connect=30.0),
        max_retries=0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=_MAX_CONCURRENT + 10,
                max_keepalive_connections=_MAX_CONCURRENT,
            ),
            timeout=httpx.Timeout(_REQUEST_TIMEOUT, connect=30.0),
        ),
    )


def _get_model() -> str:
    return os.environ.get("LLM_JUDGE_MODEL", "default")


def _encode_image(raw_bytes: bytes) -> str:
    """Base64-encode raw image bytes into a data-URI string."""
    return "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode("utf-8")


def _build_messages(
    prompt: str,
    images: Optional[list] = None,
) -> list[dict]:
    """Build the ``messages`` list for the chat API, with optional images."""
    if not images:
        return [{"role": "user", "content": prompt}]

    content: list[dict] = []
    for img in images:
        raw = img["bytes"] if isinstance(img, dict) else img
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image(raw)},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


async def llm_judge(
    student_answer: str,
    ground_truth: str,
    question: str,
    *,
    images: Optional[list] = None,
    max_retries: int = 3,
) -> float:
    """Return 1.0 if the LLM considers the answers equivalent, 0.0 otherwise.

    Parameters
    ----------
    student_answer : str
    ground_truth : str
    question : str, optional
        The original problem text, providing context for choice / semantic matching.
    images : list, optional
        Problem images. Each element is a dict with a ``"bytes"`` key (raw bytes)
        following the convention used elsewhere in the codebase (e.g. genRM).
    max_retries : int

    The concurrency of in-flight requests is capped by an asyncio.Semaphore
    (controlled via ``LLM_JUDGE_MAX_CONCURRENT``, default 64) so that the
    sglang / vLLM server is not overwhelmed when a large batch is scored
    concurrently.  Retries use exponential backoff.

    Gracefully returns 0.0 when the judge endpoint is unavailable.
    """
    try:
        client = _get_async_client()
    except Exception as e:
        logger.error("LLM judge unavailable (%s) – returning 0.0", e)
        return 0.0

    model = _get_model()
    if question is not None and '<think>' in question:
        dpsk_prompt = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the userwith the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\nNow, answer the following question: '
        question = question.replace(dpsk_prompt, '')
    prompt = JUDGE_PROMPT.format(
        question=question or "(question is not provided)",
        ground_truth=ground_truth,
        student_answer=student_answer,
    )
    messages = _build_messages(prompt, images)

    sem = _get_semaphore()
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            reply = resp.choices[0].message.content.strip().lower()
            if "yes" in reply and "no" in reply:
                logger.warning("Ambiguous LLM judge reply: %s", reply)
                return 0
            if "yes" in reply:
                return 1.0
            if "no" in reply:
                return 0.0
            logger.warning("Ambiguous LLM judge reply: %s", reply)
            return 0.0
        except Exception as e:
            logger.warning(
                "LLM judge attempt %d/%d failed: %s", attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                backoff = min(2 ** attempt, 30)
                await asyncio.sleep(backoff)

    logger.error("LLM judge exhausted %d retries – returning 0.0", max_retries)
    return 0.0


if __name__ == "__main__":
    # Test 1: _build_messages with text only
    messages_text = _build_messages("What is 2+2?")
    print("Messages (text only):", messages_text)
    assert messages_text == [{"role": "user", "content": "What is 2+2?"}]

    # Test 2: _build_messages with prompt from JUDGE_PROMPT (no images)
    prompt = JUDGE_PROMPT.format(
        question="Compute 3 * 4.",
        ground_truth="12",
        student_answer="12",
    )
    messages_judge = _build_messages(prompt)
    print("Judge messages (no images):", len(messages_judge), "message(s)")
    print("Judge messages (no images):", messages_judge)
    assert len(messages_judge) == 1 and messages_judge[0]["role"] == "user"

    # Test 3: _build_messages with fake image bytes
    fake_image = {"bytes": b"\xff\xd8\xff fake jpeg"}
    messages_with_img = _build_messages("Solve the equation.", images=[fake_image])
    print("Messages (with 1 image):", len(messages_with_img), "message(s)")
    content = messages_with_img[0]["content"]
    assert isinstance(content, list)
    assert any(
        c.get("type") == "image_url" for c in content
    ), "content should include image_url"
    assert any(c.get("type") == "text" for c in content), "content should include text"

    # Test 4: llm_judge (requires LLM_JUDGE_BASE_URL; returns 0.0 if unavailable)
    score = asyncio.run(llm_judge(
        student_answer="12",
        ground_truth="12",
        question="What is 3 * 4?",
    ))
    print("llm_judge(12, 12, question='What is 3 * 4?'):", score)
    print("Done. Set LLM_JUDGE_BASE_URL to test full API.")

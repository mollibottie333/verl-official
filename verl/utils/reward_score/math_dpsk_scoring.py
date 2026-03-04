import logging
import re
from typing import Optional

from math_verify import verify

from verl.utils.reward_score.llm_judge import llm_judge
from verl.utils.reward_score.math_dpsk_rule_and_llm import (
    is_choice_only,
    is_pure_choice,
    try_math_parse,
)

logger = logging.getLogger(__name__)


def _extract_choice(text: str) -> str:
    """Return the single choice letter (lower-cased) from *text*."""
    return re.sub(r"[^a-zA-Z]", "", text).lower()


async def score_answer_text(
    answer_text: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> tuple[float, str]:
    """Score already-extracted *answer_text* against *ground_truth*.

    Returns:
        tuple: (score, method) where score is 1.0 or 0.0 and method is one of
               ``"pure_choice"``, ``"math_verify"``, or ``"llm_judge"``.

    3-tier logic (same as math_dpsk_rule_and_llm):
      1. GT is a choice letter (A-E) -> pure-choice match or LLM judge
      2. GT parseable by math-verify -> verify; else LLM judge
      3. Otherwise -> LLM judge

    *extra_info* (optional dict) may contain ``"question"`` (str) and
    ``"images"`` (list) which are forwarded to the LLM judge for context.
    """
    answer_text = answer_text.strip()
    gt_text = ground_truth.strip()

    question = extra_info.get("question", "")
    images = extra_info.get("images", None)

    judge_kwargs = dict(images=images)

    # ---- Case 1: GT is a choice letter (A-E) ----
    if is_choice_only(gt_text):
        if is_pure_choice(answer_text):
            score = 1.0 if _extract_choice(answer_text) == gt_text.lower() else 0.0
            return score, "pure_choice"
        logger.debug("Choice GT but non-choice answer → LLM judge")
        return await llm_judge(answer_text, gt_text, question, **judge_kwargs), "llm_judge"

    # ---- Case 2: GT is parseable by math-verify ----
    gt_parsed = try_math_parse(gt_text)
    if gt_parsed is not None:
        answer_parsed = try_math_parse(answer_text)
        if answer_parsed is not None:
            try:
                if verify(answer_parsed, gt_parsed, timeout_seconds=None):
                    return 1.0, "math_verify"
            except Exception as e:
                logger.debug(
                    "math-verify error: answer=%s gt=%s err=%s",
                    answer_text, gt_text, e,
                )
        logger.debug("math-verify inconclusive → LLM judge")
        return await llm_judge(answer_text, gt_text, question, **judge_kwargs), "llm_judge"

    # ---- Case 3: fallback → LLM judge ----
    logger.debug("GT not choice / not math-parseable → LLM judge")
    return await llm_judge(answer_text, gt_text, question, **judge_kwargs), "llm_judge"

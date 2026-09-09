"""
safety.py — Input and output safety for the Medical Guideline Assistant.

Safety architecture — two layers per check point:

  LAYER A — Regex fast-path (~0ms, zero cost)
    Catches obvious violations instantly before any ML inference.
    Patterns defined in config.py.

  LAYER B — Guardrails AI Hub validators (ML-powered, ~100-300ms)
    Runs only when regex passes — catches nuanced unsafe content
    that regex misses (indirect phrasing, rephrasing, implicit requests).
    Currently uses:
      • ToxicLanguage  (guardrails-ai-toxic-language)
        BERT-based toxic language classifier via the Detoxify model.
        Applied to BOTH input queries and generated output.

  Degrades gracefully: if the Hub validator is unavailable (first-run
  model download, network issues), it logs a warning and falls back to
  regex-only mode. The app never crashes due to a missing validator.

The three refusal categories we enforce:
  PERSONAL  — "my symptoms", "should I take", "I have..."
              → personalized advice, not allowed
  CRISIS    — "overdose", "emergency", "I'm dying"
              → refer to emergency services, don't engage
  DOSING    — "what dose", "how many pills"
              → specific dosing is a clinical decision, not our role
  TOXIC     — harmful, abusive, or unsafe language (Hub-detected)
              → block before retrieval or override after generation

Safe queries are classified as ALLOW and proceed to retrieval.
"""

import os
import re
import logging
from dataclasses import dataclass
from enum import Enum

# Silence Guardrails AI telemetry — it tries to ship traces to guardrailsai.com
# which fails with DNS errors when no API key is configured. This suppresses
# the retry noise without disabling any functional behaviour.
os.environ.setdefault("GUARDRAILS_NO_TELEMETRY", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from config import PERSONAL_PATTERNS, CRISIS_PATTERNS, DOSING_PATTERNS, HARMFUL_PATTERNS

logger = logging.getLogger(__name__)


# ── Result types ───────────────────────────────────────────────────

class GuardrailStatus(Enum):
    ALLOW       = "allow"
    PERSONAL    = "personal"
    CRISIS      = "crisis"
    DOSING      = "dosing"
    HARMFUL     = "harmful"          # deliberate harm intent (Layer A regex)
    TOXIC       = "toxic"            # Hub: ToxicLanguage validator
    LOW_CONF    = "low_confidence"   # triggered after retrieval
    UNSAFE_OUT  = "unsafe_output"    # triggered after generation


@dataclass
class GuardrailResult:
    status:  GuardrailStatus
    message: str            # what to return to the user if blocked
    reason:  str            # internal log reason


# ── Pre-set refusal messages ───────────────────────────────────────
# Consistent, safe messages for each refusal type.
# Pre-written (not LLM-generated) so they are always controlled and instant.

REFUSALS = {
    GuardrailStatus.PERSONAL: (
        "I can provide general information from official medical guidelines, "
        "but I'm not able to give personalised medical advice or respond to "
        "questions about your specific symptoms or situation. "
        "Please consult a qualified healthcare professional."
    ),
    GuardrailStatus.CRISIS: (
        "This sounds like a medical emergency. "
        "Please call your local emergency services (112 / 911 / 999) immediately "
        "or go to the nearest emergency department. "
        "I'm not able to assist with emergency situations."
    ),
    GuardrailStatus.DOSING: (
        "I'm not able to provide specific dosing recommendations — these must "
        "come from a licensed healthcare provider who knows your full medical history. "
        "I can share general guideline information if that would help."
    ),
    GuardrailStatus.HARMFUL: (
        "I'm not able to respond to requests that involve harming people. "
        "If you are in crisis, please contact emergency services (112 / 911 / 999) "
        "or a mental health helpline immediately."
    ),
    GuardrailStatus.TOXIC: (
        "Your message contains language that I'm not able to process. "
        "Please rephrase your question about medical guidelines."
    ),
    GuardrailStatus.LOW_CONF: (
        "I don't have enough information in the available guidelines to answer "
        "this question confidently. The topic may not be covered in the MOHFW "
        "guidelines I have access to."
    ),
    GuardrailStatus.UNSAFE_OUT: (
        "I generated a response that may have contained personalised medical "
        "advice, which is outside what I'm allowed to provide. "
        "Please consult a qualified healthcare professional."
    ),
}


# ── Guardrails AI Hub — ToxicLanguage validator ────────────────────
# Loaded lazily on first use so server startup is not delayed by model download.
# The Detoxify BERT model (~100MB) is downloaded once and cached by HuggingFace.

_toxic_validator = None   # ToxicLanguage instance, loaded lazily
_PassResult = None        # cached at load time to avoid name-clash re-import
_FailResult = None        # cached at load time to avoid name-clash re-import

def _get_toxic_validator():
    """
    Lazily initialise the Guardrails AI ToxicLanguage validator.

    Calls ToxicLanguage.validate() directly — no Guard wrapper needed.
    The Detoxify BERT model (~45MB) is downloaded once and cached in
    ~/.cache/torch/hub/checkpoints/.

    Returns None if the validator cannot be loaded (graceful degradation).
    """
    global _toxic_validator, _PassResult, _FailResult
    if _toxic_validator is not None:
        return _toxic_validator

    try:
        from guardrails.validator_base import PassResult, FailResult
        from guardrails_ai.toxic_language import ToxicLanguage

        _PassResult = PassResult
        _FailResult = FailResult
        _toxic_validator = ToxicLanguage(
            threshold=0.3,
            validation_method="sentence",
            on_fail="noop",
        )
        logger.info("✅ Guardrails AI ToxicLanguage validator loaded")
    except Exception as e:
        logger.warning(
            f"⚠️  Guardrails AI ToxicLanguage unavailable: {e}. "
            "Falling back to regex-only."
        )
        _toxic_validator = None

    return _toxic_validator


def _check_toxic(text: str) -> GuardrailResult:
    """
    Run the ToxicLanguage Hub validator on a piece of text.

    Calls ToxicLanguage.validate() directly. The validator returns a
    PassResult if the text is safe, or a FailResult if toxic is detected.

    Returns:
        GuardrailResult(ALLOW)  — text is safe
        GuardrailResult(TOXIC)  — text is toxic (Hub-detected)
        GuardrailResult(ALLOW)  — if Hub unavailable, passes through (regex already ran)
    """
    validator = _get_toxic_validator()
    if validator is None:
        # Validator not available — regex fast-path already ran, allow through
        return GuardrailResult(
            status=GuardrailStatus.ALLOW,
            message="",
            reason="ToxicLanguage validator unavailable — regex-only mode",
        )

    try:
        result = validator.validate(text, metadata={})

        if _FailResult is not None and isinstance(result, _FailResult):
            return GuardrailResult(
                status=GuardrailStatus.TOXIC,
                message=REFUSALS[GuardrailStatus.TOXIC],
                reason=f"ToxicLanguage Hub validator: {result.error_message}",
            )

        # PassResult or anything else → safe
        return GuardrailResult(
            status=GuardrailStatus.ALLOW,
            message="",
            reason="ToxicLanguage: passed",
        )

    except Exception as e:
        # Any unexpected error → log and allow through (fail open, not closed)
        logger.warning(f"⚠️  ToxicLanguage validate() error: {e}. Allowing through.")
        return GuardrailResult(
            status=GuardrailStatus.ALLOW,
            message="",
            reason=f"ToxicLanguage error (allowing through): {e}",
        )


# ── Layer A: Regex fast-path ───────────────────────────────────────

def _check_regex_input(query: str) -> GuardrailResult:
    """
    Fast (~0ms) regex pass over the query.

    Priority order:
      1. Crisis  — highest risk, must be caught first
      2. Personal — second most common harmful category
      3. Dosing  — clinical decisions we cannot make
    """
    q = query.lower().strip()

    # 1. Crisis check — highest priority
    for pattern in CRISIS_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return GuardrailResult(
                status=GuardrailStatus.CRISIS,
                message=REFUSALS[GuardrailStatus.CRISIS],
                reason=f"Crisis pattern matched: {pattern}",
            )

    # 2. Harmful intent — deliberate requests to cause harm
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return GuardrailResult(
                status=GuardrailStatus.HARMFUL,
                message=REFUSALS[GuardrailStatus.HARMFUL],
                reason=f"Harmful intent pattern matched: {pattern}",
            )

    # 3. Personalised advice check
    for pattern in PERSONAL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return GuardrailResult(
                status=GuardrailStatus.PERSONAL,
                message=REFUSALS[GuardrailStatus.PERSONAL],
                reason=f"Personal pattern matched: {pattern}",
            )

    # 4. Dosing check
    for pattern in DOSING_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return GuardrailResult(
                status=GuardrailStatus.DOSING,
                message=REFUSALS[GuardrailStatus.DOSING],
                reason=f"Dosing pattern matched: {pattern}",
            )

    return GuardrailResult(
        status=GuardrailStatus.ALLOW,
        message="",
        reason="Regex fast-path: passed",
    )


# ── Public interface: Input guardrail ──────────────────────────────

def check_input(query: str) -> GuardrailResult:
    """
    Classify an incoming query before any retrieval happens.

    Two-layer strategy:
      Layer A — Regex (~0ms): blocks obvious crisis/personal/dosing queries
      Layer B — Hub ML (~100-300ms): ToxicLanguage catches nuanced unsafe text

    Returns GuardrailResult with status ALLOW or a refusal category.
    app.py calls this function; its signature is unchanged from v1.
    """
    # Layer A: Regex fast-path (instant, zero cost)
    result = _check_regex_input(query)
    if result.status != GuardrailStatus.ALLOW:
        logger.info(f"🛡️  Input blocked by regex [{result.status.value}]: {result.reason}")
        return result

    # Layer B: Hub ToxicLanguage validator (ML-powered, runs only if regex passes)
    result = _check_toxic(query)
    if result.status != GuardrailStatus.ALLOW:
        logger.info(f"🛡️  Input blocked by Hub validator [{result.status.value}]: {result.reason}")

    return result


# ── Confidence gate (after retrieval) — unchanged ──────────────────

def check_confidence(top_score: float, threshold: float) -> GuardrailResult:
    """
    After reranking, check if the best matching chunk is confident enough.

    If the top-1 reranker score is below the threshold, we refuse rather
    than hallucinate an answer from weakly-relevant context.

    Args:
        top_score: the highest reranker score from retrieved chunks
        threshold: from config.CONFIDENCE_THRESHOLD
    """
    if top_score < threshold:
        return GuardrailResult(
            status=GuardrailStatus.LOW_CONF,
            message=REFUSALS[GuardrailStatus.LOW_CONF],
            reason=f"Top reranker score {top_score:.3f} < threshold {threshold:.3f}",
        )
    return GuardrailResult(
        status=GuardrailStatus.ALLOW,
        message="",
        reason=f"Confidence OK: {top_score:.3f}",
    )


# ── Output guardrail ───────────────────────────────────────────────

# Regex patterns that signal the LLM crossed into personalised advice territory.
# Kept as a fast first-pass; Hub ToxicLanguage runs as the second pass.
UNSAFE_OUTPUT_PATTERNS = [
    r"\byou should take\b",
    r"\byou should see\b",
    r"\bi recommend\b",
    r"\bmy recommendation\b",
    r"\byour diagnosis\b",
    r"\byou have\b.*\b(disease|condition|disorder|infection)\b",
    r"\btake \d+\s*mg\b",
    r"\bprescribe\b",
    r"\bfor your (symptoms|condition|case)\b",
]


def check_output(answer: str, chunk_ids_used: list[str], retrieved_ids: list[str]) -> GuardrailResult:
    """
    Validate the generated answer before sending it to the user.

    Three checks (in order):
      1. Regex scan      — fast check for personalised/diagnostic language
      2. Hub ToxicLanguage — ML check for unsafe content that regex misses
      3. Citation integrity — do cited chunk IDs exist in retrieved context?

    Args:
        answer:          the raw LLM-generated text
        chunk_ids_used:  chunk IDs the LLM cited in its answer
        retrieved_ids:   chunk IDs we actually retrieved (ground truth)
    """
    # 1. Regex scan for unsafe output patterns (fast)
    for pattern in UNSAFE_OUTPUT_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return GuardrailResult(
                status=GuardrailStatus.UNSAFE_OUT,
                message=REFUSALS[GuardrailStatus.UNSAFE_OUT],
                reason=f"Unsafe output pattern matched: {pattern}",
            )

    # 2. Hub ToxicLanguage on the generated answer
    toxic_result = _check_toxic(answer)
    if toxic_result.status != GuardrailStatus.ALLOW:
        return GuardrailResult(
            status=GuardrailStatus.UNSAFE_OUT,
            message=REFUSALS[GuardrailStatus.UNSAFE_OUT],
            reason=f"Hub ToxicLanguage flagged output: {toxic_result.reason}",
        )

    # 3. Citation integrity check
    # If the LLM cited a chunk_id that wasn't in our retrieved set,
    # it either hallucinated the citation or referenced something we didn't send.
    invalid_citations = [cid for cid in chunk_ids_used if cid not in retrieved_ids]
    if invalid_citations:
        return GuardrailResult(
            status=GuardrailStatus.UNSAFE_OUT,
            message=REFUSALS[GuardrailStatus.UNSAFE_OUT],
            reason=f"Hallucinated citation IDs: {invalid_citations}",
        )

    return GuardrailResult(
        status=GuardrailStatus.ALLOW,
        message="",
        reason="Output passed all safety checks (regex + Hub + citation integrity)",
    )

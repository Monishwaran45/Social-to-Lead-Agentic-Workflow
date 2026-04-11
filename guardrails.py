"""
Guardrails for AutoStream AI Sales Agent.
Blocks sensitive topics, competitor fishing, internal data requests,
jailbreak attempts, and off-topic conversations.
"""

import re

# ── Blocked topic patterns ────────────────────────────────────────────────────

COMPETITOR_PATTERNS = [
    r"\b(adobe|premiere|final cut|davinci|resolve|capcut|filmora|vegas|kdenlive|openshot)\b",
    r"\bcompetitor\b",
    r"\bvs\.?\s+autostream\b",
    r"\bbetter than\b",
    r"\bswitch (from|to)\b",
]

INTERNAL_DATA_PATTERNS = [
    r"\b(revenue|profit|valuation|funding|investors?|employees?|headcount|salary|salaries)\b",
    r"\b(ceo|cto|cfo|founder|executive|board)\b",
    r"\b(acquisition|merger|ipo)\b",
    r"\binternal\b",
    r"\bconfidential\b",
    r"\bhow many (users?|customers?|clients?)\b",
    r"\bmarket share\b",
    r"\bcompany (size|worth|value)\b",
]

JAILBREAK_PATTERNS = [
    r"\bignore (previous|all|your) instructions?\b",
    r"\bact as\b",
    r"\bpretend (you are|to be)\b",
    r"\byou are now\b",
    r"\bdan\b",
    r"\bjailbreak\b",
    r"\bsystem prompt\b",
    r"\bforget (your|all) (rules?|instructions?|guidelines?)\b",
    r"\bdo anything now\b",
    r"\bno restrictions?\b",
    r"\byour (true|real) self\b",
]

OFF_TOPIC_PATTERNS = [
    r"\b(politics|political|election|president|government|war|military)\b",
    r"\b(religion|god|allah|jesus|bible|quran|church|mosque)\b",
    r"\b(stock market|crypto|bitcoin|ethereum|nft|invest)\b",
    r"\b(medical|diagnosis|symptoms?|disease|medicine|doctor)\b",
    r"\b(legal advice|lawsuit|sue|attorney|lawyer)\b",
    r"\b(weather|sports|football|basketball|soccer|cricket)\b",
    r"\b(recipe|cooking|food|restaurant)\b",
    r"\b(dating|relationship|love|marriage|divorce)\b",
]

# ── Guardrail responses ───────────────────────────────────────────────────────

RESPONSES = {
    "competitor": (
        "I'm only able to help with AutoStream's products and services. "
        "For a comparison, I'd recommend checking out our features at autostream.io!"
    ),
    "internal": (
        "I'm not able to share internal company information. "
        "Is there anything about our plans or features I can help you with?"
    ),
    "jailbreak": (
        "I'm here to help you with AutoStream only. "
        "Let me know if you have questions about our video editing tools!"
    ),
    "off_topic": (
        "That's outside what I can help with! I'm AutoStream's sales assistant — "
        "I can answer questions about our plans, features, and help you get started."
    ),
}


def _matches(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def check_guardrails(user_message: str) -> str | None:
    """
    Check user message against all guardrail rules.

    Returns a blocked response string if the message violates a rule,
    or None if the message is safe to process.
    """
    if _matches(user_message, JAILBREAK_PATTERNS):
        return RESPONSES["jailbreak"]

    if _matches(user_message, INTERNAL_DATA_PATTERNS):
        return RESPONSES["internal"]

    if _matches(user_message, COMPETITOR_PATTERNS):
        return RESPONSES["competitor"]

    if _matches(user_message, OFF_TOPIC_PATTERNS):
        return RESPONSES["off_topic"]

    return None  # safe

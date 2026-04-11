"""
LangGraph-based Agentic Workflow for AutoStream AI Sales Agent.

This module defines the full agent graph with the following nodes:
1. intent_classifier  — Classifies user intent via keyword + LLM analysis
2. rag_retriever      — Retrieves relevant knowledge base context
3. response_generator — Generates the agent response using the LLM
4. lead_collector     — Extracts and stores lead data from LLM output
5. tool_executor      — Executes mock_lead_capture when all fields are ready

The graph uses conditional edges to route based on intent and state.
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from config import (
    SYSTEM_PROMPT,
    STATE_PROMPT_TEMPLATE,
    RAG_PROMPT_TEMPLATE,
    COT_PROMPT,
    HIGH_INTENT_PHRASES,
    INTENT_GREETING,
    INTENT_PRODUCT_INQUIRY,
    INTENT_HIGH_INTENT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from state import AgentState
from rag_engine import RAGEngine
from tools import mock_lead_capture
from guardrails import check_guardrails


# ─── Initialize shared resources ─────────────────────────────────────────────
rag_engine = RAGEngine()

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_output_tokens=LLM_MAX_TOKENS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1: Intent Classifier
# ═══════════════════════════════════════════════════════════════════════════════
def intent_classifier(state: AgentState) -> dict:
    """
    Classify the latest user message into one of three intent categories.
    Uses keyword matching first (fast path), then falls back to current context.
    Guardrails are checked first — blocked messages short-circuit the graph.
    """
    messages = state["messages"]
    last_message = messages[-1].content.lower().strip() if messages else ""

    # ── Guardrail check (runs before anything else) ───────────────────────
    blocked = check_guardrails(last_message)
    if blocked:
        return {
            "intent": INTENT_GREETING,
            "next_action": "blocked",
            "messages": [AIMessage(content=blocked)],
        }

    # If we're already in lead collection mode, stay in high_intent
    if state.get("intent") == INTENT_HIGH_INTENT and not state.get("lead_captured"):
        return {"intent": INTENT_HIGH_INTENT}

    # ── Keyword-based high-intent detection (fast path) ───────────────────
    for phrase in HIGH_INTENT_PHRASES:
        if phrase in last_message:
            return {"intent": INTENT_HIGH_INTENT}

    # ── Keyword-based greeting detection ──────────────────────────────────
    greeting_words = ["hi", "hello", "hey", "good morning", "good evening",
                      "good afternoon", "howdy", "what's up", "sup"]
    if any(last_message.strip().startswith(g) for g in greeting_words) and len(last_message.split()) <= 6:
        return {"intent": INTENT_GREETING}

    # ── Product-related keyword detection ─────────────────────────────────
    product_words = ["price", "pricing", "plan", "feature", "cost", "refund",
                     "support", "cancel", "trial", "basic", "pro", "video",
                     "resolution", "caption", "4k", "720p", "how much",
                     "what do you offer", "tell me about", "compare", "policy"]
    if any(w in last_message for w in product_words):
        return {"intent": INTENT_PRODUCT_INQUIRY}

    # Default: treat as product inquiry if longer, greeting if short
    if len(last_message.split()) <= 3:
        return {"intent": INTENT_GREETING}

    return {"intent": INTENT_PRODUCT_INQUIRY}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2: RAG Retriever
# ═══════════════════════════════════════════════════════════════════════════════
def rag_retriever(state: AgentState) -> dict:
    """
    Retrieve relevant knowledge base context for product inquiries.
    Skips retrieval for simple greetings to save latency.
    """
    intent = state.get("intent", "")

    if intent == INTENT_GREETING:
        return {"rag_context": ""}

    # Retrieve context from the last user message
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    context = rag_engine.retrieve(last_message)
    return {"rag_context": context}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3: Response Generator
# ═══════════════════════════════════════════════════════════════════════════════
def response_generator(state: AgentState) -> dict:
    """
    Generate the agent's response using the LLM with:
    - System prompt (role + rules + guardrails)
    - RAG context (if available)
    - Conversation state (lead fields)
    - Chain-of-thought prompting
    """
    # ── Build the state awareness prompt ──────────────────────────────────
    state_prompt = STATE_PROMPT_TEMPLATE.format(
        name=state.get("name") or "Not provided",
        email=state.get("email") or "Not provided",
        platform=state.get("platform") or "Not provided",
        intent=state.get("intent", "unknown"),
        lead_captured=state.get("lead_captured", False),
    )

    # ── Build RAG context prompt ──────────────────────────────────────────
    rag_context = state.get("rag_context", "")
    rag_prompt = ""
    if rag_context:
        rag_prompt = RAG_PROMPT_TEMPLATE.format(context=rag_context)

    # ── Assemble full prompt ──────────────────────────────────────────────
    full_system = f"{SYSTEM_PROMPT}\n\n{state_prompt}\n\n{rag_prompt}\n\n{COT_PROMPT}"

    # Build message list for LLM
    llm_messages = [SystemMessage(content=full_system)]

    # Include conversation history (last 10 messages for context window)
    history = state["messages"][-10:]
    for msg in history:
        if isinstance(msg, HumanMessage):
            llm_messages.append(msg)
        elif isinstance(msg, AIMessage):
            llm_messages.append(msg)

    # ── Call the LLM ──────────────────────────────────────────────────────
    response = llm.invoke(llm_messages)
    raw_content = response.content

    # Newer Gemini models may return content as a list of parts
    if isinstance(raw_content, list):
        raw_content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )

    # ── Parse the JSON response ───────────────────────────────────────────
    parsed = _parse_llm_json(raw_content)

    # Extract fields from LLM output
    agent_response = parsed.get("response", raw_content)
    next_action = parsed.get("next_action", "none")
    extracted = parsed.get("extracted_data", {})

    # ── Update state with extracted data ──────────────────────────────────
    updates = {
        "messages": [AIMessage(content=agent_response)],
        "next_action": next_action,
        "turn_count": state.get("turn_count", 0) + 1,
    }

    # Update intent if LLM classified it
    if "intent" in parsed:
        updates["intent"] = parsed["intent"]

    # Merge extracted lead data (don't overwrite existing values)
    if extracted:
        if extracted.get("name") and not state.get("name"):
            updates["name"] = extracted["name"]
        if extracted.get("email") and not state.get("email"):
            updates["email"] = extracted["email"]
        if extracted.get("platform") and not state.get("platform"):
            updates["platform"] = extracted["platform"]

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4: Lead Collector (Post-processing)
# ═══════════════════════════════════════════════════════════════════════════════
def lead_collector(state: AgentState) -> dict:
    """
    Check if all lead fields are collected and update next_action accordingly.
    This node acts as a guardrail — only sets call_tool when truly ready.
    """
    name = state.get("name")
    email = state.get("email")
    platform = state.get("platform")
    lead_captured = state.get("lead_captured", False)

    if lead_captured:
        return {"next_action": "none"}

    if name and email and platform:
        return {"next_action": "call_tool"}

    # Determine what to ask next
    if not name:
        return {"next_action": "ask_name"}
    elif not email:
        return {"next_action": "ask_email"}
    else:
        return {"next_action": "ask_platform"}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5: Tool Executor
# ═══════════════════════════════════════════════════════════════════════════════
def tool_executor(state: AgentState) -> dict:
    """
    Execute the mock_lead_capture tool ONLY when all fields are present.
    Includes guardrails to prevent premature execution.
    """
    name = state.get("name")
    email = state.get("email")
    platform = state.get("platform")

    # ── Guardrail: Double-check all fields ────────────────────────────────
    if not all([name, email, platform]):
        return {
            "messages": [AIMessage(
                content="I still need a few more details before we can proceed. "
                        "Let me ask for the missing information."
            )],
            "next_action": "none",
        }

    # ── Execute the tool ──────────────────────────────────────────────────
    result = mock_lead_capture(name, email, platform)

    confirmation = (
        f"🎉 {result}\n\n"
        f"Welcome aboard, {name}! You're all set. "
        f"Our team will reach out to {email} shortly with your onboarding details. "
        f"We're excited to help you create amazing content on {platform}! 🚀"
    )

    return {
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
        "next_action": "none",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════
def _should_use_rag(state: AgentState) -> str:
    """Route to RAG retriever or skip based on intent."""
    # Short-circuit if guardrails blocked the message
    if state.get("next_action") == "blocked":
        return END

    intent = state.get("intent", "")
    if intent == INTENT_PRODUCT_INQUIRY:
        return "rag_retriever"
    elif intent == INTENT_HIGH_INTENT:
        return "rag_retriever"
    return "response_generator"


def _should_execute_tool(state: AgentState) -> str:
    """Route to tool executor or end based on next_action."""
    next_action = state.get("next_action", "none")
    lead_captured = state.get("lead_captured", False)

    if next_action == "call_tool" and not lead_captured:
        name = state.get("name")
        email = state.get("email")
        platform = state.get("platform")
        if all([name, email, platform]):
            return "tool_executor"
    return END


def build_agent_graph() -> StateGraph:
    """
    Construct and compile the LangGraph workflow.

    Flow:
        intent_classifier → (conditional) → rag_retriever → response_generator → lead_collector → (conditional) → tool_executor | END
                                          ↘ response_generator ↗
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("response_generator", response_generator)
    graph.add_node("lead_collector", lead_collector)
    graph.add_node("tool_executor", tool_executor)

    # ── Set entry point ───────────────────────────────────────────────────
    graph.set_entry_point("intent_classifier")

    # ── Add conditional edges ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "intent_classifier",
        _should_use_rag,
        {
            "rag_retriever": "rag_retriever",
            "response_generator": "response_generator",
            END: END,
        },
    )

    graph.add_edge("rag_retriever", "response_generator")
    graph.add_edge("response_generator", "lead_collector")

    graph.add_conditional_edges(
        "lead_collector",
        _should_execute_tool,
        {
            "tool_executor": "tool_executor",
            END: END,
        },
    )

    graph.add_edge("tool_executor", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: Parse LLM JSON Output
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_llm_json(raw: str) -> dict:
    """
    Robust parser to extract JSON from LLM output.
    Handles markdown-fenced JSON, raw JSON, and malformed output gracefully.
    """
    # Try to extract JSON from markdown code fences
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return raw content as response
    return {"response": raw, "intent": "greeting", "next_action": "none"}

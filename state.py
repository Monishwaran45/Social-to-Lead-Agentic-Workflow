"""
State management for the AutoStream AI Sales Agent.

Defines the typed state schema used by LangGraph to track conversation
context, collected lead data, and agent decisions across turns.
"""

from typing import Annotated, Optional, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    LangGraph state schema for the sales agent.

    Attributes:
        messages: Full conversation history (managed by LangGraph's add_messages reducer).
        intent: Current classified intent (greeting | product_inquiry | high_intent).
        name: Collected lead name (None until provided by user).
        email: Collected lead email (None until provided by user).
        platform: Collected lead platform (None until provided by user).
        lead_captured: Whether the lead capture tool has been called.
        next_action: The next action the agent should take.
        rag_context: Retrieved knowledge base context for the current turn.
        turn_count: Number of conversation turns elapsed.
    """
    messages: Annotated[list, add_messages]
    intent: str
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool
    next_action: str
    rag_context: str
    turn_count: int

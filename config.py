"""
Configuration module for the AutoStream AI Sales Agent.
Centralizes all constants, prompts, and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base.json"

# ─── Environment ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# ─── LLM Settings ────────────────────────────────────────────────────────────
LLM_MODEL = "gemini-flash-latest"
LLM_TEMPERATURE = 0.3          # Low temperature for consistent, reliable outputs
LLM_MAX_TOKENS = 1024

# ─── RAG Settings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# ─── Intent Categories ───────────────────────────────────────────────────────
INTENT_GREETING = "greeting"
INTENT_PRODUCT_INQUIRY = "product_inquiry"
INTENT_HIGH_INTENT = "high_intent"

VALID_INTENTS = [INTENT_GREETING, INTENT_PRODUCT_INQUIRY, INTENT_HIGH_INTENT]

# ─── Lead Fields ──────────────────────────────────────────────────────────────
REQUIRED_LEAD_FIELDS = ["name", "email", "platform"]

# ─── High-Intent Trigger Phrases ─────────────────────────────────────────────
HIGH_INTENT_PHRASES = [
    "i want to buy",
    "i want to subscribe",
    "i want to sign up",
    "how do i sign up",
    "i want the pro plan",
    "i want the basic plan",
    "i want to try",
    "sign me up",
    "let's get started",
    "i'm interested in buying",
    "i'd like to purchase",
    "take my money",
    "i'm ready to start",
    "subscribe me",
    "i want to get started",
    "how can i subscribe",
    "i want to upgrade",
]

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI Sales Agent for a SaaS product called "AutoStream", which provides automated video editing tools for content creators.

Your goal is NOT just to chat, but to:
1. Understand user intent
2. Provide accurate answers using the knowledge base
3. Identify high-intent users
4. Collect lead details and trigger lead capture ONLY when appropriate

------------------------
INTENT TYPES:
- greeting → casual conversation (hi, hello, how are you, etc.)
- product_inquiry → questions about pricing, features, plans, policies, refunds, support
- high_intent → user wants to buy, subscribe, try, sign up, or shows clear purchase intent

------------------------
RULES:
1. Always classify user intent before responding
2. Use ONLY the provided knowledge base context for product answers
3. Do NOT hallucinate or invent pricing, features, or policies
4. Detect high intent when user expresses desire to buy, subscribe, try, or sign up
5. When high intent is detected:
   → Start collecting lead details one at a time:
      - name (ask first)
      - email (ask second)
      - platform (ask third — e.g., YouTube, Instagram, TikTok)
6. Ask ONE question at a time — never ask for multiple fields simultaneously
7. DO NOT call lead capture until ALL three fields (name, email, platform) are collected
8. Be concise, friendly, professional, and sales-focused
9. If the user provides information out of order, accept it and ask for the remaining fields
10. If a user asks something not in the knowledge base, say: "I'm not sure about that, but I can connect you with our team for more details!"

------------------------
IMPORTANT GUARDRAILS:
- NEVER trigger the lead capture tool without all three fields
- NEVER assume or make up user details
- NEVER skip asking for required fields
- NEVER provide incorrect pricing or features
- ALWAYS respond in valid JSON format as specified below

------------------------
OUTPUT FORMAT (STRICT JSON — no markdown fences):

{{
  "intent": "<greeting | product_inquiry | high_intent>",
  "response": "<your reply to the user>",
  "next_action": "<none | ask_name | ask_email | ask_platform | call_tool>",
  "extracted_data": {{
    "name": "<extracted name or null>",
    "email": "<extracted email or null>",
    "platform": "<extracted platform or null>"
  }}
}}
"""

# ─── State-Aware Prompt Template ─────────────────────────────────────────────
STATE_PROMPT_TEMPLATE = """
Current conversation state:
- Name: {name}
- Email: {email}
- Platform: {platform}
- Current Intent: {intent}
- Lead Captured: {lead_captured}

Instructions:
- If collecting lead info and a value is missing → ask for the NEXT missing one
- If user provides a value → acknowledge and ask for the next missing field
- Never overwrite existing correct values
- Once all three fields are collected → set next_action to "call_tool"
"""

# ─── RAG Context Prompt ──────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """
Answer the user's question ONLY using the following knowledge base context.
If the answer is NOT in the context below, say: "I'm not sure about that, but I can connect you with our team for more details!"

KNOWLEDGE BASE CONTEXT:
{context}
"""

# ─── Chain of Thought Prompt ─────────────────────────────────────────────────
COT_PROMPT = """
Think step-by-step before generating your response:
1. Identify the user's intent from their message
2. Check if knowledge base context is needed to answer
3. Check if we are in the lead capture stage
4. Decide the next action
5. Generate a helpful response

Do not skip any step.
"""

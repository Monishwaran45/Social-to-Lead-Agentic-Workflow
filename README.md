# 🤖 Social-to-Lead Agentic Workflow — AutoStream AI Sales Agent

> A production-grade conversational AI agent built with **LangGraph + Gemini 1.5 Flash** that converts social media conversations into qualified business leads for **AutoStream** — an automated video editing SaaS for content creators.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?logo=langchain)
![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-orange?logo=google)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture-explanation)
- [Project Structure](#-project-structure)
- [Setup & Run](#-setup--run-locally)
- [Demo Conversation](#-demo-conversation-flow)
- [WhatsApp Deployment](#-whatsapp-deployment-guide)
- [Tech Stack](#-tech-stack)

---

## ✨ Features

| Capability | Description |
|---|---|
| 🧠 **Intent Detection** | Classifies user messages into `greeting`, `product_inquiry`, or `high_intent` using keyword matching + LLM reasoning |
| 📚 **RAG-Powered Knowledge** | Retrieves accurate product/pricing info from a FAISS vector store using sentence-transformer embeddings |
| 🎯 **Lead Qualification** | Detects purchase intent and systematically collects name, email, and platform |
| 🔧 **Tool Execution** | Calls `mock_lead_capture()` only when all three fields are collected (with guardrails) |
| 💾 **Stateful Memory** | Maintains full conversation context across 5-6+ turns using LangGraph's typed state |
| 🛡️ **Guardrails** | Prevents premature tool calls, hallucinated data, and incorrect pricing |

---

## 🏗️ Architecture Explanation

### Why LangGraph?

I chose **LangGraph** over alternatives like AutoGen for several key reasons:

1. **Explicit Control Flow**: LangGraph provides a graph-based workflow where each node (intent classification, RAG retrieval, response generation, lead collection, tool execution) is a discrete, testable unit. This makes the agent's reasoning transparent and debuggable — critical for production sales agents.

2. **Typed State Management**: LangGraph's `TypedDict` state schema gives us compile-time safety and clear documentation of what data flows through the system. The state persists across turns naturally, maintaining the conversation's `name`, `email`, `platform`, `intent`, and `lead_captured` fields without external storage.

3. **Conditional Routing**: The graph's conditional edges enable smart routing — greeting messages skip RAG retrieval entirely, while high-intent messages route through lead collection logic. This reduces unnecessary LLM calls and latency.

4. **Guardrailed Tool Execution**: The `lead_collector` node acts as a safety layer before `tool_executor`, double-checking that all required fields exist before triggering the CRM mock. This prevents the common failure mode of premature tool calls.

The architecture follows a **ReAct-inspired pattern** where the LLM reasons about intent and state, then the graph takes structured actions based on that reasoning. The structured JSON output format ensures reliable parsing, while the chain-of-thought prompting ensures the LLM considers all factors before responding.

```
┌──────────────┐     ┌───────────────┐     ┌─────────────────────┐     ┌─────────────────┐     ┌────────────────┐
│   Intent     │────▶│ RAG Retriever │────▶│ Response Generator  │────▶│ Lead Collector  │────▶│ Tool Executor  │
│  Classifier  │     │ (conditional) │     │   (LLM + Prompts)   │     │  (guardrails)   │     │ (conditional)  │
└──────────────┘     └───────────────┘     └─────────────────────┘     └─────────────────┘     └────────────────┘
       │                                                                                               │
       │                    Greeting → skip RAG                              Missing fields → END      │
       └─────────────────────────────▶ response_generator                                              │
                                                                              All fields → tool_executor
                                                                                                       ▼
                                                                                                    ✅ END
```

---

## 📁 Project Structure

```
Social-to-Lead Agentic Workflow/
├── main.py              # CLI entry point — conversation loop & rich UI
├── agent.py             # LangGraph workflow — 5 nodes + conditional routing
├── config.py            # All prompts, constants, and settings
├── state.py             # LangGraph typed state schema
├── rag_engine.py        # FAISS vector store + sentence-transformer RAG
├── tools.py             # mock_lead_capture() tool
├── knowledge_base.json  # Product data — plans, pricing, policies, FAQ
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

---

## 🚀 Setup & Run Locally

### Prerequisites

- **Python 3.9+** installed
- **Google Gemini API key** (free tier works) — [Get one here](https://aistudio.google.com/apikey)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/social-to-lead-agentic-workflow.git
cd social-to-lead-agentic-workflow
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GOOGLE_API_KEY=your_actual_key_here
```

### Step 5: Run the Agent

```bash
python main.py
```

### Optional: Enable Debug Mode

```bash
# Shows internal state after each turn
set DEBUG=true    # Windows
export DEBUG=true # macOS/Linux
python main.py
```

---

## 💬 Demo Conversation Flow

```
You → Hi, tell me about your pricing.

🤖 AutoStream Agent:
Hey there! 👋 Here's what we offer:

📦 Basic Plan — $29/month
  • 10 videos/month
  • 720p resolution

🚀 Pro Plan — $79/month
  • Unlimited videos
  • 4K resolution
  • AI-powered captions
  • 24/7 dedicated support

Would you like to know more about any plan?

──────────────────────────────────────

You → That sounds great, I want to try the Pro plan for my YouTube channel.

🤖 AutoStream Agent:
Awesome choice! The Pro plan is perfect for YouTube creators. 🎬
Let me get you set up — what's your name?

──────────────────────────────────────

You → I'm Alex

🤖 AutoStream Agent:
Nice to meet you, Alex! What's your email address so we can send you the onboarding details?

──────────────────────────────────────

You → alex@example.com

🤖 AutoStream Agent:
Great! And which platform do you primarily create content for? (YouTube, Instagram, TikTok, etc.)

──────────────────────────────────────

You → YouTube

🎯 CRM Lead Entry:
┌──────────────────────────────────┐
│ ✅ Lead Captured Successfully!   │
│                                  │
│ Name:     Alex                   │
│ Email:    alex@example.com       │
│ Platform: YouTube                │
│ Status:   new_lead               │
│ Source:   autostream_ai_agent    │
└──────────────────────────────────┘

🤖 AutoStream Agent:
🎉 Welcome aboard, Alex! You're all set.
Our team will reach out to alex@example.com shortly with your onboarding details.
We're excited to help you create amazing content on YouTube! 🚀
```

---

## 📱 WhatsApp Deployment Guide

To deploy this agent on WhatsApp, I would use the following architecture:

### Architecture Overview

```
User ←→ WhatsApp ←→ Meta Cloud API ←→ Webhook Server (Flask/FastAPI) ←→ LangGraph Agent
```

### Implementation Steps

1. **Meta Business Account Setup**
   - Register on [Meta for Developers](https://developers.facebook.com/)
   - Create a WhatsApp Business App and get a phone number
   - Generate a permanent access token

2. **Webhook Server (Flask/FastAPI)**
   - Create a `/webhook` endpoint that:
     - Handles Meta's **verification challenge** (`GET` request with `hub.verify_token`)
     - Receives **incoming messages** via `POST` requests
     - Extracts the user's phone number and message text
     - Routes the message to the LangGraph agent
     - Sends the agent's response back via the WhatsApp Cloud API

   ```python
   @app.post("/webhook")
   async def webhook(request: Request):
       data = await request.json()
       message = extract_message(data)
       
       # Load/create user session state (keyed by phone number)
       state = session_store.get(message.phone_number, default_state())
       
       # Run the LangGraph agent
       state["messages"].append(HumanMessage(content=message.text))
       result = agent.invoke(state)
       
       # Save updated state
       session_store.set(message.phone_number, result)
       
       # Send response via WhatsApp Cloud API
       send_whatsapp_message(message.phone_number, result["messages"][-1].content)
   ```

3. **Session Management**
   - Use **Redis** or **DynamoDB** keyed by phone number to maintain per-user state across turns
   - This replaces the in-memory state dict used in the CLI version

4. **Deployment**
   - Host on **AWS Lambda + API Gateway** (serverless) or **Railway/Render** (always-on)
   - Use **ngrok** for local development/testing
   - Set the webhook URL in Meta Developer Dashboard

5. **Security**
   - Validate the `X-Hub-Signature-256` header on incoming webhooks
   - Rate-limit per phone number to prevent abuse
   - Store tokens in environment variables, never in code

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Agent Framework** | LangGraph (LangChain) |
| **LLM** | Google Gemini 1.5 Flash |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS (CPU) |
| **State Management** | LangGraph TypedDict State |
| **Terminal UI** | Rich |
| **Prompt Engineering** | ReAct + Role + Structured Output + Guardrails |

---

## 📄 License

This project is built as part of the **ServiceHive / Inflx** ML Intern Assignment.

---

<p align="center">
  <strong>Built with 🧠 by a Senior ML Engineer</strong><br>
  <em>ReAct-based structured prompting with guardrails and JSON outputs for reliable reasoning, controlled tool execution, and stateful conversation handling.</em>
</p>

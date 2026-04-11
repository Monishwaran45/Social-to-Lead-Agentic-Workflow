"""Quick end-to-end test for the AutoStream agent + Supabase lead capture."""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent import build_agent_graph

print("\n=== AutoStream Agent - E2E Test ===\n")

agent = build_agent_graph()

state = {
    "messages": [],
    "intent": "",
    "name": None,
    "email": None,
    "platform": None,
    "lead_captured": False,
    "next_action": "none",
    "rag_context": "",
    "turn_count": 0,
}

turns = [
    "Hi there!",
    "I want to subscribe to AutoStream",
    "My name is John Doe",
    "john@example.com",
    "YouTube",
]

for user_msg in turns:
    print(f"You    : {user_msg}")
    state["messages"].append(HumanMessage(content=user_msg))

    result = agent.invoke(state)
    state.update(result)

    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            print(f"Agent  : {msg.content}")
            break

    print(f"[state] name={state.get('name')} | email={state.get('email')} | platform={state.get('platform')} | captured={state.get('lead_captured')}")
    print()

print("=== Test Complete ===")

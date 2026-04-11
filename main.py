"""
Main entry point for the AutoStream AI Sales Agent.

Provides an interactive CLI loop with rich terminal output.
Initializes the LangGraph agent and manages the conversation lifecycle.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Validate API key ─────────────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY is not set.")
    print("   Please create a .env file with your Gemini API key.")
    print("   See .env.example for reference.")
    sys.exit(1)

from agent import build_agent_graph

console = Console()

# ─── ASCII Banner ─────────────────────────────────────────────────────────────
BANNER = """
[bold cyan]
 █████╗ ██╗   ██╗████████╗ ██████╗ ███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║
███████║██║   ██║   ██║   ██║   ██║███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║
██╔══██║██║   ██║   ██║   ██║   ██║╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║
██║  ██║╚██████╔╝   ██║   ╚██████╔╝███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝
[/bold cyan]
[bold white]       🤖 AI Sales Agent · Powered by LangGraph + Gemini[/bold white]
"""


def display_welcome():
    """Show the welcome banner and instructions."""
    console.print(BANNER)
    console.print(Panel(
        "[bold]Welcome to AutoStream AI Sales Agent![/bold]\n\n"
        "I can help you with:\n"
        "  💰  Pricing & plan information\n"
        "  🎬  Product features & capabilities\n"
        "  📋  Company policies\n"
        "  🚀  Signing up for a plan\n\n"
        "[dim]Type [bold]quit[/bold] or [bold]exit[/bold] to end the session.[/dim]",
        border_style="cyan",
        title="[bold]Getting Started[/bold]",
        subtitle="[dim]AutoStream · Automated Video Editing for Creators[/dim]",
    ))
    console.print()


def run_agent():
    """Main conversation loop."""
    display_welcome()

    # ── Build the LangGraph agent ─────────────────────────────────────────
    console.print("[dim]Initializing agent...[/dim]")
    agent = build_agent_graph()
    console.print("[green]✅ Agent is ready![/green]\n")

    # ── Initialize conversation state ─────────────────────────────────────
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

    # ── Conversation loop ─────────────────────────────────────────────────
    while True:
        try:
            user_input = console.input("[bold cyan]You → [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            console.print(Panel(
                "[bold]Thanks for chatting with AutoStream! 👋[/bold]\n"
                "Have a great day!",
                border_style="cyan",
            ))
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # ── Run the agent graph ───────────────────────────────────────────
        try:
            result = agent.invoke(state)

            # Update state from result
            state.update(result)

            # Display the agent's response
            if result["messages"]:
                last_ai_msg = None
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "content") and msg.content:
                        last_ai_msg = msg
                        break

                if last_ai_msg:
                    console.print()
                    console.print(Panel(
                        last_ai_msg.content,
                        title="[bold magenta]🤖 AutoStream Agent[/bold magenta]",
                        border_style="magenta",
                        padding=(1, 2),
                    ))
                    console.print()

            # ── Debug: Show current state (optional) ──────────────────────
            _show_debug_state(state)

        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            console.print("[dim]Please try again or rephrase your question.[/dim]\n")


def _show_debug_state(state: dict):
    """Display current conversation state for debugging."""
    debug = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    if not debug:
        return

    console.print(Panel(
        f"[dim]Intent:[/dim]    {state.get('intent', 'N/A')}\n"
        f"[dim]Name:[/dim]     {state.get('name', 'N/A')}\n"
        f"[dim]Email:[/dim]    {state.get('email', 'N/A')}\n"
        f"[dim]Platform:[/dim] {state.get('platform', 'N/A')}\n"
        f"[dim]Captured:[/dim] {state.get('lead_captured', False)}\n"
        f"[dim]Action:[/dim]   {state.get('next_action', 'N/A')}\n"
        f"[dim]Turn:[/dim]     {state.get('turn_count', 0)}",
        title="[dim]Debug State[/dim]",
        border_style="dim",
    ))


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_agent()

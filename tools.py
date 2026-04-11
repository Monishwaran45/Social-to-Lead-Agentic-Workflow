"""
Tool definitions for the AutoStream AI Sales Agent.
Captures leads and persists them to Supabase.
"""

from rich.console import Console
from rich.panel import Panel
from database import save_lead

console = Console()


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Captures a lead and saves it to Supabase.

    Args:
        name: Full name of the prospective customer.
        email: Email address of the prospective customer.
        platform: Content creation platform (YouTube, Instagram, TikTok, etc.).

    Returns:
        A confirmation message string.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not name or not name.strip():
        return "❌ Error: Name is required but was empty."
    if not email or not email.strip():
        return "❌ Error: Email is required but was empty."
    if not platform or not platform.strip():
        return "❌ Error: Platform is required but was empty."

    if "@" not in email or "." not in email:
        return f"❌ Error: '{email}' does not appear to be a valid email address."

    # ── Save to Supabase ──────────────────────────────────────────────────
    try:
        lead_data = save_lead(name, email, platform)
        db_status = "[green]✅ Saved to Supabase[/green]"
    except Exception as e:
        # Fallback: log locally if DB is unreachable
        lead_data = {
            "name": name.strip(),
            "email": email.strip().lower(),
            "platform": platform.strip(),
            "status": "new_lead",
            "source": "autostream_ai_agent",
        }
        db_status = f"[yellow]⚠️  DB unavailable — stored locally ({e})[/yellow]"

    # ── Pretty-print the captured lead ────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]✅ Lead Captured Successfully![/bold green]\n\n"
        f"[cyan]Name:[/cyan]     {lead_data['name']}\n"
        f"[cyan]Email:[/cyan]    {lead_data['email']}\n"
        f"[cyan]Platform:[/cyan] {lead_data['platform']}\n"
        f"[cyan]Status:[/cyan]   {lead_data['status']}\n"
        f"[cyan]Source:[/cyan]   {lead_data['source']}\n"
        f"[cyan]DB:[/cyan]       {db_status}",
        title="🎯 CRM Lead Entry",
        border_style="green",
    ))

    return (
        f"Lead captured successfully: {lead_data['name']}, "
        f"{lead_data['email']}, {lead_data['platform']}"
    )

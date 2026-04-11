"""
Database module for AutoStream AI Sales Agent.
Handles lead storage in Supabase via the supabase-py client.
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


def _get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def save_lead(name: str, email: str, platform: str) -> dict:
    """
    Insert a lead into the Supabase 'leads' table.
    Returns the saved lead data dict.
    """
    import uuid

    lead_data = {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "email": email.strip().lower(),
        "platform": platform.strip(),
        "status": "new_lead",
        "source": "autostream_ai_agent",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    client = _get_client()
    response = client.table("leads").insert(lead_data).execute()

    if hasattr(response, "error") and response.error:
        raise Exception(f"Supabase insert error: {response.error}")

    return lead_data

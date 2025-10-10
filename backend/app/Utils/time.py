from typing import Any, Dict
from datetime import datetime, timezone

def add_timestamps(doc: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    doc.setdefault("created_at", now)
    doc.setdefault("updated_at", now)
    return doc
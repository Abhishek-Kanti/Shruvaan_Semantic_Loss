import hashlib
import json
from datetime import datetime, timezone

class AuditLogger:
    def __init__(self):
        self.logs = []
        self.last_hash = None

    def log(self, component, event, details):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "event": event,
            "details": details,
            "prev_hash": self.last_hash
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["hash"] = entry_hash

        self.logs.append(entry)
        self.last_hash = entry_hash
        return entry

    def export(self, filepath="audit_log.json"):
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)

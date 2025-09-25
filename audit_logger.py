# import hashlib
# import json
# from datetime import datetime, timezone

# class AuditLogger:
#     def __init__(self):
#         self.logs = []
#         self.last_hash = None

#     def log(self, component, event, details):
#         entry = {
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "component": component,
#             "event": event,
#             "details": details,
#             "prev_hash": self.last_hash
#         }
#         entry_str = json.dumps(entry, sort_keys=True)
#         entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
#         entry["hash"] = entry_hash

#         self.logs.append(entry)
#         self.last_hash = entry_hash
#         return entry

#     def export(self, filepath="audit_log.json"):
#         with open(filepath, "w") as f:
#             json.dump(self.logs, f, indent=2)

#     # ---- New helper ----
#     def get_theta_for_packet(self, hctx_hex=None, salt_hex=None):
#         """
#         Look through logs for the latest Cryptor->Encrypt entry matching hctx+salt,
#         and return the theta vector if present. Otherwise return None.
#         """
#         for entry in reversed(self.logs):
#             if entry["component"] == "Cryptor" and entry["event"] == "Encrypt":
#                 det = entry.get("details", {})
#                 if hctx_hex and det.get("hctx") != hctx_hex:
#                     continue
#                 if salt_hex and det.get("salt") != salt_hex:
#                     continue
#                 return det.get("theta")
#         return None



import hashlib
import json
from datetime import datetime, timezone
import os

class AuditLogger:
    def __init__(self, filepath: str = "audit_log.json"):
        self.logs = []
        self.last_hash = None
        self.filepath = filepath
        # load existing file (if any) to memory so we don't lose history
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.logs = json.load(f)
                    if self.logs:
                        self.last_hash = self.logs[-1].get("hash")
            except Exception:
                # ignore malformed file; start fresh
                self.logs = []
                self.last_hash = None

    def _atomic_write(self, logs):
        # simple atomic write using temp file then rename
        tmp = self.filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(logs, f, indent=2)
        os.replace(tmp, self.filepath)

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

        # persist immediately so other processes can read
        try:
            self._atomic_write(self.logs)
        except Exception:
            # do not fail user code for logger file write issues
            pass

        return entry

    # ---- New helper ----
    def get_theta_for_packet(self, hctx_hex=None, salt_hex=None):
        """
        Look through logs (in-memory first, then on-disk) for the latest Cryptor->Encrypt
        entry matching hctx+salt, and return the theta vector if present. Otherwise return None.
        """
        # check in-memory logs first
        for entry in reversed(self.logs):
            if entry["component"] == "Cryptor" and entry["event"] == "Encrypt":
                det = entry.get("details", {})
                if hctx_hex and det.get("hctx") != hctx_hex:
                    continue
                if salt_hex and det.get("salt") != salt_hex:
                    continue
                return det.get("theta")

        # fallback: try loading from disk (if file exists)
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r") as f:
                    logs_disk = json.load(f)
                for entry in reversed(logs_disk):
                    if entry.get("component") == "Cryptor" and entry.get("event") == "Encrypt":
                        det = entry.get("details", {})
                        if hctx_hex and det.get("hctx") != hctx_hex:
                            continue
                        if salt_hex and det.get("salt") != salt_hex:
                            continue
                        return det.get("theta")
        except Exception:
            # swallow file-reading exceptions; return None below
            pass

        return None

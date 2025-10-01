import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

class CryptoHistoryLogger:
    def __init__(self, history_file: str = "crypto_history.json"):
        self.history_file = history_file
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w") as f:
                json.dump([], f)

    def log_pair(self, enc_packet: Dict[str, Any], dec_payload: Dict[str, Any]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "encrypted": enc_packet,
            "decrypted": dec_payload
        }
        history = self._load_all()
        history.append(record)
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def load_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        history = self._load_all()
        return history[-n:]

    def _load_all(self):
        with open(self.history_file, "r") as f:
            return json.load(f)

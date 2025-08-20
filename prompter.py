import google.generativeai as genai
from datetime import datetime, timezone
import os
import json
from dotenv import load_dotenv
from audit_logger import AuditLogger

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Prompter:
    def __init__(self, api_key=None, default_role="TellerAgent", default_policy="≤10000", logger: AuditLogger = None):
        self.default_role = default_role
        self.default_policy = default_policy
        self.logger = logger or AuditLogger()

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not set. Pass api_key or set GEMINI_API_KEY env var.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def normalize_instruction(self, instruction: str):
        prompt = f"""
        You are a financial instruction normalizer.
        Take the user instruction and convert it into strict JSON.

        Supported actions: transfer, balance_check, account_mgmt.

        Instruction: {instruction}

        Output JSON with these keys:
        - action (string: transfer | balance_check | account_mgmt | unknown)
        - amount (number or null)
        - currency (string or null)
        - from_account (string or null)
        - to_account (string or null)
        """
        resp = self.model.generate_content(prompt)
        raw = resp.text.strip()

        # Strip markdown fences if Gemini wraps output
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            normalized = json.loads(raw)
        except Exception:
            raise ValueError(f"Gemini returned invalid JSON: {raw}")

        # Add metadata
        normalized["policy"] = self.default_policy
        normalized["role"] = self.default_role
        normalized["epoch"] = datetime.now(timezone.utc).isoformat()
        normalized["raw_instruction"] = instruction.strip()

        # Audit log entry
        self.logger.log(
            component="Prompter",
            event="Normalize",
            details={
                "raw_instruction": instruction.strip(),
                "normalized": normalized,
            }
        )

        return normalized


# --- Quick test ---
# if __name__ == "__main__":
#     logger = AuditLogger()
#     p = Prompter(api_key=GEMINI_API_KEY, logger=logger)

#     tests = [
#         "transfer $9000 from 56789 to 45678",
#         "please transfer 9,000 to 45678 from 56789",
#         "check balance of account 56789",
#         "close account 11111",
#         "enquiry remaining balance 123456789"
#     ]
#     for t in tests:
#         out = p.normalize_instruction(t)
#         print(f"{t} → {out}\n")

#     # Export full chain once
#     logger.export("audit_log.json")
#     print("\nAudit log exported to audit_log.json")

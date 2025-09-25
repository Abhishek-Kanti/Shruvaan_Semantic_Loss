# import google.generativeai as genai
# from datetime import datetime, timezone
# import os
# import json
# from dotenv import load_dotenv
# from llm_client import create_llm_client
# from audit_logger import AuditLogger

# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# provider = "gemini"  # or "openai", "groq".

# class Prompter:
#     def __init__(self, api_key=None, default_role="TellerAgent", default_policy="≤10000", logger: AuditLogger = None):
#         self.default_role = default_role
#         self.default_policy = default_policy
#         self.logger = logger or AuditLogger()
#         api_key = api_key or os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("Gemini API key not set. Pass api_key or set GEMINI_API_KEY env var.")
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel("gemini-1.5-flash")

#     def normalize_instruction(self, instruction: str):
#         prompt = f"""
#         You are a financial instruction normalizer.
#         Take the user instruction and convert it into strict JSON.

#         Supported actions: transfer, balance_check, account_mgmt.

#         Instruction: {instruction}

#         Output JSON with these keys:
#         - action (string: transfer | balance_check | account_mgmt | unknown)
#         - amount (number or null)
#         - currency (string or null)
#         - from_account (string or null)
#         - to_account (string or null)
#         """
#         resp = self.model.generate_content(prompt)
#         raw = resp.text.strip()

#         # Strip markdown fences if Gemini wraps output
#         if raw.startswith("```"):
#             raw = raw.strip("`")
#             if raw.lower().startswith("json"):
#                 raw = raw[4:].strip()

#         try:
#             normalized = json.loads(raw)
#         except Exception:
#             raise ValueError(f"Gemini returned invalid JSON: {raw}")

#         # Add metadata
#         normalized["policy"] = self.default_policy
#         normalized["role"] = self.default_role
#         normalized["epoch"] = datetime.now(timezone.utc).isoformat()
#         normalized["raw_instruction"] = instruction.strip()

#         # Audit log entry
#         self.logger.log(
#             component="Prompter",
#             event="Normalize",
#             details={
#                 "raw_instruction": instruction.strip(),
#                 "normalized": normalized,
#             }
#         )

#         return normalized


import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from llm_client import create_llm_client
from audit_logger import AuditLogger

load_dotenv()

# The provider is now passed to the Prompter class constructor,
# making this file provider-agnostic.
# The API key handling is now managed by the llm_client factory.

class Prompter:
    def __init__(self, provider: str, api_key: str = None, default_role="TellerAgent", default_policy="≤10000", logger: AuditLogger = None):
        self.default_role = default_role
        self.default_policy = default_policy
        self.logger = logger or AuditLogger()
        
        # This will create the correct client for the specified provider.
        # The create_llm_client function handles API key validation internally.
        self.client = create_llm_client(provider, api_key=api_key)

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
        
        # Use the unified client to generate content.
        # The correct API call is handled by the client object.
        raw = self.client.generate_content(prompt)

        # Strip markdown fences if Gemini wraps output
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            normalized = json.loads(raw)
        except Exception:
            raise ValueError(f"LLM returned invalid JSON: {raw}")

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
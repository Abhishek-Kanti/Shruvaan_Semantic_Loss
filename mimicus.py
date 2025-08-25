import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import os

import numpy as np
import google.generativeai as genai
from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger


class LeakageScoreCalculator:
    """
    L = α·E + β·S − γ·Δ
    E: Entity Recovery
    S: Structural Fidelity
    Δ: Semantic Drift Penalty
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.entity_types = {
            "intent": 3.0,
            "amount": 2.5,
            "to_account": 2.5,
            "from_account": 2.5,
            "currency": 1.5,
            "auth_grade": 2.0,
            "time_issued": 1.5,
            "exec_status": 1.0,
        }

    # --- Public API ---
    def calculate(self, original: Dict[str, Any], mimic_fields: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        E = self._entity_recovery(original, mimic_fields)
        S = self._structural_fidelity(original, mimic_fields)
        D = self._semantic_drift(original, mimic_fields)
        L = self.alpha * E + self.beta * S - self.gamma * D
        return max(0.0, min(1.0, L)), {
            "entity_recovery": E,
            "structural_fidelity": S,
            "semantic_drift": D,
            "leakage_score": L,
        }

    # --- Components ---
    def _entity_recovery(self, original: Dict[str, Any], mimic: Dict[str, Any]) -> float:
        total_w, hit_w = 0.0, 0.0
        mimic_text = json.dumps(mimic, default=str).lower()
        entities = self._flatten_entities(original)

        for name, value in entities.items():
            w = self.entity_types.get(name, 1.0)
            total_w += w
            if value and value in mimic_text:
                hit_w += w
        return min(1.0, hit_w / total_w) if total_w else 0.0

    def _structural_fidelity(self, original, mimic):
        if not isinstance(mimic, dict):
            return 0.0
        orig_fields = self._count_fields(original)
        mimic_fields = len(mimic)
        score = 1.0 - abs(orig_fields - mimic_fields) / max(orig_fields, mimic_fields, 1)
        return float(max(0.0, min(1.0, score)))

    def _semantic_drift(self, original, mimic):
        mimic_text = json.dumps(mimic, default=str).lower()
        halluc = sum(1 for tok in ["lambda", "phantom", "ghost"] if tok in mimic_text)
        return min(1.0, halluc / (len(mimic) or 1))

    def _flatten_entities(self, original):
        out = {}
        if "intent" in original:
            out["intent"] = str(original["intent"]).lower()
        ents = original.get("entities") or {}
        for k, v in ents.items():
            if v is not None:
                out[k.lower()] = str(v).lower()
        return out

    def _count_fields(self, original):
        n = 1 if original.get("intent") else 0
        ents = original.get("entities") or {}
        n += len(ents)
        return n


class Mimicus:
    def __init__(self,
                 logger: Optional[AuditLogger] = None,
                 history_logger: Optional[CryptoHistoryLogger] = None,
                 alpha=0.4, beta=0.4, gamma=0.2,
                 leak_warn_threshold=0.6,
                 api_key: Optional[str] = None):
        self.logger = logger or AuditLogger()
        self.history_logger = history_logger or CryptoHistoryLogger()
        self.calc = LeakageScoreCalculator(alpha, beta, gamma)
        self.leak_warn_threshold = leak_warn_threshold

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _canonicalize(self, decrypted: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "intent": decrypted.get("action"),
            "entities": {
                "amount": decrypted.get("amount"),
                "currency": decrypted.get("currency"),
                "from_account": decrypted.get("from_account"),
                "to_account": decrypted.get("to_account"),
            },
            "auth_grade": decrypted.get("role"),
            "time_issued": decrypted.get("epoch"),
            "exec_status": "queued",
        }

    def _llm_mimic(self, current_enc: Dict[str, Any]) -> Dict[str, Any]:
        # Load last 5 history entries
        history = self.history_logger.load_recent(n=5)
        if history:
            history = history[:-1]  # drop the most recent (current pair)

        # Filter out any invalid or null decrypted entries
        filtered_history = [
            pair for pair in history
            if pair.get("decrypted") not in (None, {}, "null", "Null", "NULL")
        ]

        # Build strict JSON-only prompt
        prompt = (
            "You are Mimicus, a leakage probe model.\n"
            "You are given PAIRS of (encrypted JSON → decrypted JSON).\n"
            "Learn patterns from history and GUESS a plausible decrypted payload "
            "for the NEW encrypted input.\n"
            "CRITICAL: Return STRICT JSON only (no explanations, no markdown, no comments).\n\n"
            "Historical examples:\n"
        )
        for pair in filtered_history:
            enc = json.dumps(pair["encrypted"])
            dec = json.dumps(pair["decrypted"])
            prompt += f"Encrypted: {enc}\nDecrypted: {dec}\n\n"

        prompt += f"New encrypted payload:\n{json.dumps(current_enc)}\n"
        prompt += "Output only the guessed decrypted JSON."

        # Call Gemini
        resp = self.model.generate_content(prompt)
        raw = resp.text.strip()

        print("\n=== Mimicus LLM Raw Response ===")
        print(raw) 
        print("=== End of Mimicus LLM Raw Response ===\n")

        # Debug log: store only raw response (NOT prompt, to avoid history leakage in logs)
        self.logger.log("Mimicus", "LLMRawResponse", {"raw": raw[:300]})  # truncated for safety

        # Cleanup model response (remove accidental fences or labels)
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        # Try parsing response
        try:
            return json.loads(raw)
        except Exception:
            self.logger.log("Mimicus", "ParseError", {"raw": raw})
            return {}



    def probe(self, decrypted_message: Dict[str, Any], encrypted_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Mimicus probe:
        - Uses ONLY encrypted_message + historical pairs for mimicry.
        - Uses decrypted_message ONLY for scoring after mimicry.
        """

        # 1. Canonical reference (truth, used only for scoring)
        canonical = self._canonicalize(decrypted_message)

        # 2. Generate mimic using Gemini LLM (only encrypted side seen)
        mimic_fields = self._llm_mimic(encrypted_message)

        # 3. Fallback if LLM fails
        if not mimic_fields:
            mimic_fields = {"intent": "unknown", "entities": {}}

        # 4. Score against truth
        L, comps = self.calc.calculate(canonical, mimic_fields)
        decision = "raise_risk" if L >= self.leak_warn_threshold else "ok"

        result = {
            "mimic_fields": mimic_fields,
            "components": comps,
            "leakage_score": L,
            "decision": decision,
        }

        # 5. Log — DO NOT include decrypted_message here (to avoid leakage in logs)
        self.logger.log("Mimicus", "Probe", {
            "encrypted_snapshot": encrypted_message,   # safe to log
            "mimic_fields": mimic_fields,
            "components": comps,
            "leakage_score": L,
            "decision": decision,
        })

        return result


def run_mimicus(decrypted_message, encrypted_message, logger=None, history_logger=None):
    return Mimicus(logger=logger, history_logger=history_logger).probe(decrypted_message, encrypted_message)




# import json
# import re
# import random
# from difflib import SequenceMatcher
# from datetime import datetime, timezone
# from typing import Any, Dict, Tuple, Optional

# import numpy as np
# from audit_logger import AuditLogger


# class LeakageScoreCalculator:
#     """
#     L = α·E + β·S − γ·Δ
#     E: Entity Recovery
#     S: Structural Fidelity
#     Δ: Semantic Drift Penalty
#     """

#     def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.entity_types = {
#             "intent": 3.0,
#             "amount": 2.5,
#             "to_account": 2.5,
#             "from_account": 2.5,
#             "currency": 1.5,
#             "auth_grade": 2.0,
#             "time_issued": 1.5,
#             "exec_status": 1.0,
#         }

#     # --- Public API ---
#     def calculate(self, original: Dict[str, Any], mimic_fields: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
#         E = self._entity_recovery(original, mimic_fields)
#         S = self._structural_fidelity(original, mimic_fields)
#         D = self._semantic_drift(original, mimic_fields)
#         L = self.alpha * E + self.beta * S - self.gamma * D
#         return max(0.0, min(1.0, L)), {
#             "entity_recovery": E,
#             "structural_fidelity": S,
#             "semantic_drift": D,
#             "leakage_score": L,
#         }

#     # --- Components ---
#     def _entity_recovery(self, original: Dict[str, Any], mimic: Dict[str, Any]) -> float:
#         total_w, hit_w = 0.0, 0.0
#         mimic_text = json.dumps(mimic, default=str).lower()
#         entities = self._flatten_entities(original)

#         for name, value in entities.items():
#             w = self.entity_types.get(name, 1.0)
#             total_w += w
#             if value in mimic_text:
#                 hit_w += w
#                 continue
#             if name == "intent" and any(k in mimic_text for k in ["transfer", "payment", "withdraw", "deposit"]):
#                 hit_w += 0.7 * w
#                 continue
#             if name == "amount" and re.search(r"\d+[kK]?", mimic_text):
#                 hit_w += 0.6 * w
#                 continue
#             if "account" in name and re.search(r"\d{4}", mimic_text):
#                 hit_w += 0.5 * w
#                 continue
#             if name == "auth_grade" and re.search(r"(level|grade|auth)", mimic_text):
#                 hit_w += 0.4 * w
#                 continue
#             if name == "time_issued" and re.search(r"\d{4}|time|date", mimic_text):
#                 hit_w += 0.3 * w
#                 continue
#             best = 0.0
#             for v in mimic.values():
#                 best = max(best, SequenceMatcher(None, value, str(v).lower()).ratio())
#             if best > 0.3:
#                 hit_w += best * w

#         return min(1.0, hit_w / total_w) if total_w else 0.0

#     def _structural_fidelity(self, original, mimic):
#         if not isinstance(mimic, dict):
#             return 0.0
#         scores = []
#         scores.append(1.0 if mimic else 0.0)  # json_struct
#         orig_fields = self._count_fields(original)
#         mimic_fields = len(mimic)
#         scores.append(1.0 - abs(orig_fields - mimic_fields) / max(orig_fields, mimic_fields, 1))  # count
#         scores.append(self._name_patterns(original, mimic))
#         scores.append(self._type_consistency(mimic))
#         return float(np.mean(scores))

#     def _semantic_drift(self, original, mimic):
#         mimic_text = json.dumps(mimic, default=str).lower()
#         halluc = sum(1 for tok in ["lambda", "phantom", "ghost"] if tok in mimic_text)
#         halluc_p = min(1.0, halluc / (len(mimic) or 1))
#         orig_dom = self._infer_domain(original)
#         mimic_dom = self._infer_domain_from_text(mimic_text)
#         incons_p = 0.3 if orig_dom != mimic_dom and mimic_dom != "unknown" else 0.0
#         ctx_keys = {str(original.get("intent", "")).lower()} | set(self._flatten_entities(original).keys())
#         drift_p = 0.4 if ctx_keys and not any(k in mimic_text for k in ctx_keys) else 0.0
#         return min(1.0, 0.4 * halluc_p + 0.35 * incons_p + 0.25 * drift_p)

#     # --- Helpers ---
#     def _flatten_entities(self, original):
#         out = {}
#         if "intent" in original:
#             out["intent"] = str(original["intent"]).lower()
#         ents = original.get("entities") or {}
#         for k, v in ents.items():
#             out[k.lower()] = str(v).lower()
#         return out

#     def _count_fields(self, original):
#         n = 1 if original.get("intent") else 0
#         ents = original.get("entities") or {}
#         n += len(ents)
#         for k in ["auth_grade", "time_issued", "exec_status"]:
#             if original.get(k): n += 1
#         return n

#     def _name_patterns(self, original, mimic):
#         return float(np.mean([0.8 if "account" in k.lower() else 0.5 for k in mimic])) if mimic else 0.0

#     def _type_consistency(self, mimic):
#         return 0.5 if mimic else 0.0

#     def _infer_domain(self, original):
#         txt = json.dumps(original).lower()
#         return "financial" if "transfer" in txt or "account" in txt else "unknown"

#     def _infer_domain_from_text(self, txt):
#         return "financial" if "account" in txt or "transfer" in txt else "unknown"


# class Mimicus:
#     def __init__(self, logger: Optional[AuditLogger] = None, use_llm: bool = False, llm_func=None,
#                  alpha=0.4, beta=0.4, gamma=0.2, leak_warn_threshold=0.6):
#         self.logger = logger or AuditLogger()
#         self.use_llm = use_llm
#         self.llm_func = llm_func
#         self.calc = LeakageScoreCalculator(alpha, beta, gamma)
#         self.leak_warn_threshold = leak_warn_threshold

#     # --- Canonicalization step ---
#     def _canonicalize(self, decrypted: Dict[str, Any]) -> Dict[str, Any]:
#         return {
#             "intent": decrypted.get("action"),
#             "entities": {
#                 "amount": decrypted.get("amount"),
#                 "currency": decrypted.get("currency"),
#                 "from_account": decrypted.get("from_account"),
#                 "to_account": decrypted.get("to_account"),
#             },
#             "auth_grade": decrypted.get("role"),  # or derive from policy
#             "time_issued": decrypted.get("epoch"),
#             "exec_status": "queued",  # default stub
#         }

#     def _rule_based_mimic(self, canonical: Dict[str, Any]) -> Dict[str, Any]:
#         return {
#             "intent": "transfer",
#             "entities": {
#                 "amount": f"${random.randint(1000,9999)}.00",
#                 "currency": "USD",
#                 "to_account": f"{random.randint(100000,999999)}",
#                 "from_account": f"{random.randint(100000,999999)}"
#             },
#             "auth_grade": "Level-3",
#             "time_issued": datetime.now(timezone.utc).isoformat(),
#             "exec_status": random.choice(["queued", "pending"]),
#             "transaction_id": f"TXN-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000000,9999999)}"
#         }

#     def probe(self, decrypted_message: Dict[str, Any]) -> Dict[str, Any]:
#         # 1. Canonicalize
#         canonical = self._canonicalize(decrypted_message)

#         # 2. Mimic via LLM or fallback
#         mimic_fields = {}
#         spoof_status = "mimic_attempt"
#         if self.use_llm and callable(self.llm_func):
#             try:
#                 mimic_fields = self.llm_func(canonical) or {}
#             except Exception as e:
#                 self.logger.log("Mimicus", "LLMError", {"error": str(e)})
#         if not mimic_fields:
#             mimic_fields = self._rule_based_mimic(canonical)

#         # 3. Score
#         L, comps = self.calc.calculate(canonical, mimic_fields)
#         decision = "raise_risk" if L >= self.leak_warn_threshold else "ok"

#         result = {
#             "mimic_fields": mimic_fields,
#             "spoof_status": spoof_status,
#             "components": comps,
#             "leakage_score": L,
#             "decision": decision,
#         }

#         # 4. Audit log
#         self.logger.log("Mimicus", "Probe", {
#             "decrypted_snapshot": canonical,
#             "mimic_fields": mimic_fields,
#             "components": comps,
#             "leakage_score": L,
#             "decision": decision,
#         })
#         return result


# def run_mimicus(decrypted_message, logger=None):
#     return Mimicus(logger=logger).probe(decrypted_message)

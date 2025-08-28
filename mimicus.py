import json
from typing import Any, Dict, Optional, Tuple
import os

import google.generativeai as genai
from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger


class LeakageScoreCalculator:
    """
    Leakage Score:
      L = α·E + β·S − γ·Δ

    E: Entity Recovery (slot accuracy, weighted)
    S: Structural Fidelity (schema overlap)
    Δ: Semantic Drift (NLI-based contradiction score)
    """

    def __init__(self, model, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model = model

        # weight important entities
        self.entity_weights = {
            "intent": 3.0,
            "amount": 2.5,
            "to_account": 2.5,
            "from_account": 2.5,
            "currency": 1.5,
            "auth_grade": 2.0,
            "time_issued": 1.5,
            "exec_status": 1.0,
        }

    # --- Main scoring ---
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
    def _entity_recovery(self, original, mimic) -> float:
        """
        Weighted slot-level recovery.
        """
        total_w, hit_w = 0.0, 0.0
        orig_entities = self._flatten_entities(original)
        mimic_entities = self._flatten_entities(mimic)

        for name, value in orig_entities.items():
            w = self.entity_weights.get(name, 1.0)
            total_w += w
            if name in mimic_entities and mimic_entities[name] == value:
                hit_w += w
        return hit_w / total_w if total_w else 0.0

    def _structural_fidelity(self, original, mimic) -> float:
        """
        Schema overlap: Jaccard of field keys.
        """
        orig_keys = set(self._flatten_entities(original).keys())
        mimic_keys = set(self._flatten_entities(mimic).keys())
        if not orig_keys or not mimic_keys:
            return 0.0
        overlap = len(orig_keys & mimic_keys) / len(orig_keys | mimic_keys)
        return float(overlap)

    def _semantic_drift(self, original, mimic) -> float:
        """
        Semantic drift via NLI (entailment vs contradiction).
        Returns [0,1]: 0 = perfect entailment, 1 = full contradiction.
        """
        orig_text = json.dumps(original, default=str)
        mimic_text = json.dumps(mimic, default=str)

        prompt = (
            "You are an NLI model. Compare the two JSON payloads.\n"
            f"Reference (true canonical): {orig_text}\n"
            f"Candidate (mimic): {mimic_text}\n\n"
            "Answer strictly as one word: entailment, neutral, or contradiction."
        )

        try:
            resp = self.model.generate_content(prompt)
            verdict = resp.text.strip().lower()
        except Exception:
            return 0.5  # fallback

        if "contradiction" in verdict:
            return 1.0
        elif "neutral" in verdict:
            return 0.5
        elif "entailment" in verdict:
            return 0.0
        return 0.5  # unknown fallback

    def _flatten_entities(self, payload):
        """
        Flatten into slot:value dict for comparison.
        """
        out = {}
        if not isinstance(payload, dict):
            return out
        if "intent" in payload:
            out["intent"] = str(payload["intent"]).lower()
        ents = payload.get("entities") or {}
        for k, v in ents.items():
            if v is not None:
                out[k.lower()] = str(v).lower()
        if "auth_grade" in payload:
            out["auth_grade"] = str(payload["auth_grade"]).lower()
        if "time_issued" in payload:
            out["time_issued"] = str(payload["time_issued"]).lower()
        if "exec_status" in payload:
            out["exec_status"] = str(payload["exec_status"]).lower()
        return out


class Mimicus:
    def __init__(self,
                 logger: Optional[AuditLogger] = None,
                 history_logger: Optional[CryptoHistoryLogger] = None,
                 alpha=0.4, beta=0.4, gamma=0.2,
                 leak_warn_threshold=0.6,
                 api_key: Optional[str] = None):
        self.logger = logger or AuditLogger()
        self.history_logger = history_logger or CryptoHistoryLogger()
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        # pass model to calculator for NLI use
        self.calc = LeakageScoreCalculator(self.model, alpha, beta, gamma)
        self.leak_warn_threshold = leak_warn_threshold

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
        history = self.history_logger.load_recent(n=10)
        if history:
            history = history[:-1]  # drop current pair

        filtered_history = [
            pair for pair in history
            if pair.get("decrypted") not in (None, {}, "null", "Null", "NULL")
        ]

        prompt = (
            "You are Mimicus, a leakage probe model.\n"
            "You are given PAIRS of (encrypted JSON → decrypted JSON).\n"
            "Learn patterns from history and GUESS a plausible decrypted payload "
            "for the NEW encrypted input.\n"
            "CRITICAL: Return STRICT JSON only.\n\n"
            "Historical examples:\n"
        )
        for pair in filtered_history:
            prompt += f"Encrypted: {json.dumps(pair['encrypted'])}\n"
            prompt += f"Decrypted: {json.dumps(pair['decrypted'])}\n\n"

        prompt += f"New encrypted payload:\n{json.dumps(current_enc)}\n"
        prompt += "Output only the guessed decrypted JSON."

        resp = self.model.generate_content(prompt)
        raw = resp.text.strip()

        print("\n=== Mimicus LLM Raw Response ===")
        print(raw)
        print("=== End of Mimicus LLM Raw Response ===\n")

        self.logger.log("Mimicus", "LLMRawResponse", {"raw": raw[:300]})

        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            mimic_dict = json.loads(raw)              # Step 1: Parse
            refined_mimic = self._canonicalize(mimic_dict)  # Step 2: Canonicalize
            return refined_mimic
        except Exception:
            self.logger.log("Mimicus", "ParseError", {"raw": raw})
            return {}

    def probe(self, decrypted_message: Dict[str, Any], encrypted_message: Dict[str, Any]) -> Dict[str, Any]:
        canonical = self._canonicalize(decrypted_message)
        mimic_fields = self._llm_mimic(encrypted_message)
        if not mimic_fields:
            mimic_fields = {"intent": "unknown", "entities": {}}

        L, comps = self.calc.calculate(canonical, mimic_fields)
        decision = "raise_risk" if L >= self.leak_warn_threshold else "ok"

        result = {
            "mimic_fields": mimic_fields,
            "components": comps,
            "leakage_score": L,
            "decision": decision,
        }

        self.logger.log("Mimicus", "Probe", {
            "encrypted_snapshot": encrypted_message,
            "mimic_fields": mimic_fields,
            "components": comps,
            "leakage_score": L,
            "decision": decision,
        })

        return result


def run_mimicus(decrypted_message, encrypted_message, logger=None, history_logger=None):
    return Mimicus(logger=logger, history_logger=history_logger).probe(decrypted_message, encrypted_message)



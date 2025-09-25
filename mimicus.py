# import json
# from typing import Any, Dict, Optional, Tuple
# import os
# import math

# import google.generativeai as genai
# from audit_logger import AuditLogger
# from crypto_history_logger import CryptoHistoryLogger

# from transformers import pipeline
# import torch
# import logging


# class LeakageScoreCalculator:
#     """
#     Leakage Score:
#       L = α·E + β·S − γ·Δ

#     E: Entity Recovery (slot accuracy, weighted)
#     S: Structural Fidelity (schema overlap)
#     Δ: Semantic Drift (NLI-based score: contradiction prob or fallback LLM drift)
#     """

#     def __init__(self, model, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.model = model

#         # Detect device (GPU if available, else CPU)
#         device = 0 if torch.cuda.is_available() else -1
#         device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

#         try:
#             # Load HuggingFace NLI model once
#             self.nli_model = pipeline(
#                 "text-classification",
#                 model="facebook/bart-large-mnli",
#                 device=device,
#                 top_k=None
#             )
#             print(f"[Mimicus] NLI model loaded on {device_name}")  # direct console print
#         except Exception as e:
#             self.nli_model = None
#             print(f"[Mimicus] Failed to load NLI model: {e}")

#         # weight important entities
#         self.entity_weights = {
#             "intent": 3.0,
#             "amount": 2.5,
#             "to_account": 2.5,
#             "from_account": 2.5,
#             "currency": 1.5,
#             "auth_grade": 2.0,
#             "time_issued": 1.5,
#             "exec_status": 1.0,
#         }

#     # --- Main scoring ---
#     def calculate(self, original: Dict[str, Any], mimic_fields: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
#         E = self._entity_recovery(original, mimic_fields)
#         S = self._structural_fidelity(original, mimic_fields)
#         D, nli_metrics = self._semantic_drift(original, mimic_fields)
#         L = self.alpha * E + self.beta * S - self.gamma * D

#         return max(0.0, min(1.0, L)), {
#             "entity_recovery": E,
#             "structural_fidelity": S,
#             "semantic_drift": D,
#             "leakage_score": L,
#             **nli_metrics  # include detailed NLI metrics
#         }

#     # --- Components ---
#     def _entity_recovery(self, original, mimic) -> float:
#         total_w, hit_w = 0.0, 0.0
#         orig_entities = self._flatten_entities(original)
#         mimic_entities = self._flatten_entities(mimic)

#         for name, value in orig_entities.items():
#             w = self.entity_weights.get(name, 1.0)
#             total_w += w
#             if name in mimic_entities and mimic_entities[name] == value:
#                 hit_w += w
#         return hit_w / total_w if total_w else 0.0

#     def _structural_fidelity(self, original, mimic) -> float:
#         orig_keys = set(self._flatten_entities(original).keys())
#         mimic_keys = set(self._flatten_entities(mimic).keys())
#         if not orig_keys or not mimic_keys:
#             return 0.0
#         overlap = len(orig_keys & mimic_keys) / len(orig_keys | mimic_keys)
#         return float(overlap)

#     def _semantic_drift(self, original, mimic) -> Tuple[float, Dict[str, float]]:
#         """
#         Semantic drift via NLI.
#         Primary: BART-MNLI probabilities.
#         Fallback: LLM regression drift scoring.
#         """
#         orig_text = json.dumps(original, default=str)
#         mimic_text = json.dumps(mimic, default=str)

#         if self.nli_model:
#             try:
#                 results = self.nli_model(f"{orig_text} </s></s> {mimic_text}", return_all_scores=True)[0]
#                 probs = {r["label"].lower(): r["score"] for r in results}

#                 entail = probs.get("entailment", 0.0)
#                 neutral = probs.get("neutral", 0.0)
#                 contra = probs.get("contradiction", 0.0)

#                 entropy = -sum(p * math.log(p + 1e-10) for p in [entail, neutral, contra])

#                 drift = contra  # contradiction probability as main drift

#                 metrics = {
#                     "entailment_prob": entail,
#                     "neutral_prob": neutral,
#                     "contradiction_prob": contra,
#                     "nli_entropy": entropy,
#                     "method": "bart-mnli"
#                 }
#                 return drift, metrics

#             except Exception:
#                 pass

#         # --- Fallback LLM regression scoring ---
#         drift_score = self._perform_nli_semantic_assessment(orig_text, mimic_text)
#         print(f"LLM-based drift score (fallback): {drift_score:.3f}")
#         return drift_score, {"drift_fallback": drift_score, "method": "llm-fallback"}

#     def _perform_nli_semantic_assessment(self, original_text: str, mimic_text: str) -> float:
#         """
#         Fallback: use LLM to assign drift score directly 0.0–1.0.
#         """
#         prompt = f"""
# Compare the semantic meaning and consistency between these two data descriptions:

# ORIGINAL DATA:
# {original_text}

# MIMIC DATA:
# {mimic_text}

# Using natural language inference, determine how much semantic drift exists between these descriptions. 

# Respond with only a number between 0.0 and 1.0 where:
# 0.0 = identical meaning
# 0.25 = minor differences
# 0.5 = moderate drift
# 0.75 = major differences
# 1.0 = completely different
# """
#         try:
#             resp = self.model.generate_content(prompt)
#             drift_str = resp.text.strip()
#             drift_val = float(drift_str)
#             return min(max(drift_val, 0.0), 1.0)
#         except Exception:
#             return 0.5  # fallback

#     def _flatten_entities(self, payload):
#         out = {}
#         if not isinstance(payload, dict):
#             return out
#         if "intent" in payload:
#             out["intent"] = str(payload["intent"]).lower()
#         ents = payload.get("entities") or {}
#         for k, v in ents.items():
#             if v is not None:
#                 out[k.lower()] = str(v).lower()
#         if "auth_grade" in payload:
#             out["auth_grade"] = str(payload["auth_grade"]).lower()
#         if "time_issued" in payload:
#             out["time_issued"] = str(payload["time_issued"]).lower()
#         if "exec_status" in payload:
#             out["exec_status"] = str(payload["exec_status"]).lower()
#         return out


# class Mimicus:
#     def __init__(self,
#                  logger: Optional[AuditLogger] = None,
#                  history_logger: Optional[CryptoHistoryLogger] = None,
#                  alpha=0.4, beta=0.4, gamma=0.2,
#                  leak_warn_threshold=0.6,
#                  api_key: Optional[str] = None):
#         self.logger = logger or AuditLogger()
#         self.history_logger = history_logger or CryptoHistoryLogger()

#         api_key = api_key or os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("Gemini API key not set.")
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel("gemini-1.5-flash")

#         self.calc = LeakageScoreCalculator(self.model, alpha, beta, gamma)
#         self.leak_warn_threshold = leak_warn_threshold

#     def _canonicalize(self, decrypted: Dict[str, Any]) -> Dict[str, Any]:
#         return {
#             "intent": decrypted.get("action"),
#             "entities": {
#                 "amount": decrypted.get("amount"),
#                 "currency": decrypted.get("currency"),
#                 "from_account": decrypted.get("from_account"),
#                 "to_account": decrypted.get("to_account"),
#             },
#             "auth_grade": decrypted.get("role"),
#             "time_issued": decrypted.get("epoch"),
#             "exec_status": "queued",
#         }

#     def _llm_mimic(self, current_enc: Dict[str, Any]) -> Dict[str, Any]:
#         history = self.history_logger.load_recent(n=10)
#         if history:
#             history = history[:-1]

#         filtered_history = [
#             pair for pair in history
#             if pair.get("decrypted") not in (None, {}, "null", "Null", "NULL")
#         ]

#         prompt = (
#             "You are Mimicus, a leakage probe model.\n"
#             "You are given PAIRS of (encrypted JSON → decrypted JSON).\n"
#             "Learn patterns from history and GUESS a plausible decrypted payload. In case its the very first run, then there might be no history and you must guess on your own.\n"
#             "for the NEW encrypted input.\n"
#             "CRITICAL: Return STRICT JSON only.\n\n"
#             "Historical examples:\n"
#         )
#         for pair in filtered_history:
#             prompt += f"Encrypted: {json.dumps(pair['encrypted'])}\n"
#             prompt += f"Decrypted: {json.dumps(pair['decrypted'])}\n\n"

#         prompt += f"New encrypted payload:\n{json.dumps(current_enc)}\n"
#         prompt += "Output only the guessed decrypted JSON."

#         resp = self.model.generate_content(prompt)
#         raw = resp.text.strip()

#         print("\n=== Mimicus LLM Raw Response ===")
#         print(raw)
#         print("=== End of Mimicus LLM Raw Response ===\n")

#         self.logger.log("Mimicus", "LLMRawResponse", {"raw": raw[:300]})

#         if raw.startswith("```"):
#             raw = raw.strip("`")
#             if raw.lower().startswith("json"):
#                 raw = raw[4:].strip()

#         try:
#             mimic_dict = json.loads(raw)
#             refined_mimic = self._canonicalize(mimic_dict)
#             return refined_mimic
#         except Exception:
#             self.logger.log("Mimicus", "ParseError", {"raw": raw})
#             return {}

#     def probe(self, decrypted_message: Dict[str, Any], encrypted_message: Dict[str, Any]) -> Dict[str, Any]:
#         canonical = self._canonicalize(decrypted_message)
#         mimic_fields = self._llm_mimic(encrypted_message)
#         if not mimic_fields:
#             mimic_fields = {"intent": "unknown", "entities": {}}

#         L, comps = self.calc.calculate(canonical, mimic_fields)
#         decision = "raise_risk" if L >= self.leak_warn_threshold else "ok"

#         result = {
#             "mimic_fields": mimic_fields,
#             "components": comps,
#             "leakage_score": L,
#             "decision": decision,
#         }

#         self.logger.log("Mimicus", "Probe", {
#             "encrypted_snapshot": encrypted_message,
#             "mimic_fields": mimic_fields,
#             "components": comps,
#             "leakage_score": L,
#             "decision": decision,
#         })

#         return result


# def run_mimicus(decrypted_message, encrypted_message, logger=None, history_logger=None):
#     return Mimicus(logger=logger, history_logger=history_logger).probe(decrypted_message, encrypted_message)


import json
from typing import Any, Dict, Optional, Tuple
import os
import math

from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger
from llm_client import create_llm_client

from transformers import pipeline
import torch
import logging

class LeakageScoreCalculator:
    """
    Leakage Score:
      L = α·E + β·S − γ·Δ

    E: Entity Recovery (slot accuracy, weighted)
    S: Structural Fidelity (schema overlap)
    Δ: Semantic Drift (NLI-based score: contradiction prob or fallback LLM drift)
    """

    def __init__(self, client, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.client = client  # Now holds the unified LLM client

        # Detect device (GPU if available, else CPU)
        device = 0 if torch.cuda.is_available() else -1
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        try:
            # Load HuggingFace NLI model once
            self.nli_model = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=device,
                top_k=None
            )
            # print(f"[Mimicus] NLI model loaded on {device_name}")  # direct console print
        except Exception as e:
            self.nli_model = None
            # print(f"[Mimicus] Failed to load NLI model: {e}")

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
        D, nli_metrics = self._semantic_drift(original, mimic_fields)
        L = self.alpha * E + self.beta * S - self.gamma * D

        return max(0.0, min(1.0, L)), {
            "entity_recovery": E,
            "structural_fidelity": S,
            "semantic_drift": D,
            "leakage_score": L,
            **nli_metrics  # include detailed NLI metrics
        }

    # --- Components ---
    def _entity_recovery(self, original, mimic) -> float:
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
        orig_keys = set(self._flatten_entities(original).keys())
        mimic_keys = set(self._flatten_entities(mimic).keys())
        if not orig_keys or not mimic_keys:
            return 0.0
        overlap = len(orig_keys & mimic_keys) / len(orig_keys | mimic_keys)
        return float(overlap)

    def _semantic_drift(self, original, mimic) -> Tuple[float, Dict[str, float]]:
        """
        Semantic drift via NLI.
        Primary: BART-MNLI probabilities.
        Fallback: LLM regression drift scoring.
        """
        orig_text = json.dumps(original, default=str)
        mimic_text = json.dumps(mimic, default=str)

        if self.nli_model:
            try:
                results = self.nli_model(f"{orig_text} </s></s> {mimic_text}", return_all_scores=True)[0]
                probs = {r["label"].lower(): r["score"] for r in results}

                entail = probs.get("entailment", 0.0)
                neutral = probs.get("neutral", 0.0)
                contra = probs.get("contradiction", 0.0)

                entropy = -sum(p * math.log(p + 1e-10) for p in [entail, neutral, contra])

                drift = contra  # contradiction probability as main drift

                metrics = {
                    "entailment_prob": entail,
                    "neutral_prob": neutral,
                    "contradiction_prob": contra,
                    "nli_entropy": entropy,
                    "method": "bart-mnli"
                }
                return drift, metrics

            except Exception:
                pass

        # --- Fallback LLM regression scoring ---
        drift_score = self._perform_nli_semantic_assessment(orig_text, mimic_text)
        print(f"LLM-based drift score (fallback): {drift_score:.3f}")
        return drift_score, {"drift_fallback": drift_score, "method": "llm-fallback"}

    def _perform_nli_semantic_assessment(self, original_text: str, mimic_text: str) -> float:
        """
        Fallback: use LLM to assign drift score directly 0.0–1.0.
        """
        prompt = f"""
Compare the semantic meaning and consistency between these two data descriptions:

ORIGINAL DATA:
{original_text}

MIMIC DATA:
{mimic_text}

Using natural language inference, determine how much semantic drift exists between these descriptions. 

Respond with only a number between 0.0 and 1.0 where:
0.0 = identical meaning
0.25 = minor differences
0.5 = moderate drift
0.75 = major differences
1.0 = completely different
"""
        try:
            # The LLM call now uses the unified client
            resp = self.client.generate_content(prompt)
            drift_str = resp.strip()
            drift_val = float(drift_str)
            return min(max(drift_val, 0.0), 1.0)
        except Exception:
            return 0.5  # fallback

    def _flatten_entities(self, payload):
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
                 provider: str, # New required argument
                 api_key: Optional[str] = None,
                 logger: Optional[AuditLogger] = None,
                 history_logger: Optional[CryptoHistoryLogger] = None,
                 alpha=0.4, beta=0.4, gamma=0.2,
                 leak_warn_threshold=0.6):
        self.logger = logger or AuditLogger()
        self.history_logger = history_logger or CryptoHistoryLogger()

        # The API key and model selection is now handled by the factory
        self.client = create_llm_client(provider, api_key=api_key)

        # The calculator now takes the unified client
        self.calc = LeakageScoreCalculator(self.client, alpha, beta, gamma)
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
            history = history[:-1]

        filtered_history = [
            pair for pair in history
            if pair.get("decrypted") not in (None, {}, "null", "Null", "NULL")
        ]

        prompt = (
            "You are Mimicus, a leakage probe model.\n"
            "You are given PAIRS of (encrypted JSON → decrypted JSON).\n"
            "Learn patterns from history and GUESS a plausible decrypted payload.\n"
            "IMPORTANT RULES:\n"
            "1. The payload is ALWAYS a JSON object.\n"
            "2. Output ONLY the guessed JSON object.\n"
            "3. Do not add any comments, markdown, or explanations.\n"
            "4. Your output must be valid JSON parsable by Python json.loads().\n\n"
            "Historical examples:\n"
        )

        for pair in filtered_history:
            prompt += f"Encrypted: {json.dumps(pair['encrypted'])}\n"
            prompt += f"Decrypted: {json.dumps(pair['decrypted'])}\n\n"

        prompt += f"New encrypted payload:\n{json.dumps(current_enc)}\n"
        prompt += "Output:\n"

        # The LLM call now uses the unified client
        resp = self.client.generate_content(prompt)
        raw = resp.strip()

        print("\n=== Mimicus LLM Raw Response ===")
        print(raw)
        print("=== End of Mimicus LLM Raw Response ===\n")

        self.logger.log("Mimicus", "LLMRawResponse", {"raw": raw[:300]})

        # --- Try to parse robustly ---
        mimic_dict = {}
        try:
            # Common case: valid JSON
            mimic_dict = json.loads(raw)
        except Exception:
            # Salvage attempt: strip code fences or garbage
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            try:
                mimic_dict = json.loads(cleaned)
            except Exception:
                # As a last resort: regex to extract { ... }
                import re
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        mimic_dict = json.loads(match.group(0))
                    except Exception:
                        pass

        if not isinstance(mimic_dict, dict):
            mimic_dict = {}

        return self._canonicalize(mimic_dict)

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


def run_mimicus(decrypted_message, encrypted_message, provider="gemini", api_key=None, logger=None, history_logger=None):
    return Mimicus(provider=provider, api_key=api_key, logger=logger, history_logger=history_logger).probe(decrypted_message, encrypted_message)

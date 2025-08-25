import hashlib
import json
import base64
from typing import Any, Dict, Union, Optional, List
from cryptography.fernet import Fernet, InvalidToken
from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger

JsonObj = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class Decryptor:
    """
    Inverse of Cryptor:
    - Rebuild Fernet from HKP
    - Decrypt every key and every leaf value
    - Restore original types via json.loads()
    - Verify PoP
    """

    def __init__(self, logger: Optional[AuditLogger] = None,
                 history_logger: Optional[CryptoHistoryLogger] = None):
        self.logger = logger or AuditLogger()
        self.history_logger = history_logger   # shared logger injected here

    def _hash_hex(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str) -> bytes:
        material = (hctx_hex + salt_hex).encode("utf-8")
        return hashlib.sha256(material).digest()

    def _to_canonical(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _dec_str(self, cipher: Fernet, s: str) -> str:
        return cipher.decrypt(s.encode("utf-8")).decode("utf-8")

    def _decrypt_structure(self, cipher: Fernet, obj: JsonObj) -> JsonObj:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for ek, v in obj.items():
                plain_key = self._dec_str(cipher, ek)
                out[plain_key] = self._decrypt_structure(cipher, v)
            return out

        if isinstance(obj, list):
            return [self._decrypt_structure(cipher, v) for v in obj]

        if isinstance(obj, str):
            try:
                val = self._dec_str(cipher, obj)
                # try to restore type
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return val
            except InvalidToken:
                return obj

        return obj

    def decrypt(self, packet: Dict[str, Any], *,
                hctx_hex: Optional[str] = None,
                salt_hex: Optional[str] = None) -> Dict[str, Any]:

        enc_map = packet.get("enc_map")
        pop = packet.get("pop")
        metadata = packet.get("metadata", {}) or {}

        role = metadata.get("role", "")
        epoch = metadata.get("epoch", "")
        kdf = metadata.get("kdf") or {}

        hctx_hex = hctx_hex or kdf.get("hctx")
        salt_hex = salt_hex or kdf.get("salt")

        if not (hctx_hex and salt_hex):
            raise ValueError("Missing HKP KDF material")

        fernet_key = base64.urlsafe_b64encode(self._hkp_key_bytes(hctx_hex, salt_hex))
        cipher = Fernet(fernet_key)

        plaintext_obj = self._decrypt_structure(cipher, enc_map)
        if not isinstance(plaintext_obj, dict):
            raise ValueError("Decryption produced non-dict root")

        canonical_plain = self._to_canonical(plaintext_obj)
        expected_pop = self._hash_hex(canonical_plain + role + epoch)
        if expected_pop != pop:
            self.logger.log("Decryptor", "PoPMismatch", {
                "expected": expected_pop, "got": pop, "role": role, "epoch": epoch
            })
            raise ValueError("PoP verification failed")
        
        self.history_logger.log_pair(packet["enc_map"], plaintext_obj)

        self.logger.log("Decryptor", "Decrypt", {
            "role": role, "epoch": epoch, "ok": True,
            "field_count": len(plaintext_obj)
        })
        return plaintext_obj



# import hashlib
# import json
# import base64
# from cryptography.fernet import Fernet, InvalidToken
# from datetime import datetime, timezone
# from typing import Optional, Dict, Any

# from audit_logger import AuditLogger
# from policy import PolicyStore


# # ------------------------------
# # Helpers (must match Cryptor exactly)
# # ------------------------------
# def _hash(data: str) -> str:
#     return hashlib.sha256(data.encode()).hexdigest()

# def _fernet_key_from_material(hctx_hex: str, salt_hex: str) -> bytes:
#     """
#     Cryptor did:
#         hkp_key = sha256(hctx + salt).digest()
#         fernet_key = base64.urlsafe_b64encode(hkp_key)
#     This reproduces the same derivation.
#     """
#     material = (hctx_hex + salt_hex).encode()
#     hkp_key_bytes = hashlib.sha256(material).digest()
#     return base64.urlsafe_b64encode(hkp_key_bytes)


# # ------------------------------
# # Decryptor
# # ------------------------------
# class Decryptor:
#     def __init__(self, logger: AuditLogger, policy_store: PolicyStore):
#         if logger is None:
#             raise ValueError("A shared AuditLogger instance must be provided to Decryptor")
#         self.logger = logger
#         self.policy_store = policy_store

#     def _find_encrypt_log_by_pop(self, pop_hex: str) -> Optional[Dict[str, Any]]:
#         for entry in self.logger.logs:
#             if entry.get("component") == "Cryptor" and entry.get("event") == "Encrypt":
#                 details = entry.get("details", {})
#                 if details.get("pop") == pop_hex:
#                     return entry
#         return None

#     def decrypt(self, encrypted_packet: Dict[str, Any], expected_role: Optional[str] = None) -> Dict[str, Any]:
#         if "ciphertext" not in encrypted_packet or "pop" not in encrypted_packet or "metadata" not in encrypted_packet:
#             raise ValueError("Malformed encrypted_packet; missing ciphertext/pop/metadata")

#         pop_hex = encrypted_packet["pop"]
#         metadata = encrypted_packet["metadata"]
#         role = metadata.get("role")
#         epoch = metadata.get("epoch")

#         # Role check
#         if expected_role and role != expected_role:
#             self.logger.log(
#                 component="Decryptor",
#                 event="DecryptAttempt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "role_mismatch",
#                     "expected_role": expected_role,
#                     "observed_role": role,
#                     "pop": pop_hex
#                 }
#             )
#             raise ValueError(f"Role mismatch: expected {expected_role} but packet claims {role}")

#         # 1) Find corresponding encryption log (trusted hctx)
#         encrypt_log = self._find_encrypt_log_by_pop(pop_hex)
#         if not encrypt_log:
#             self.logger.log(
#                 component="Decryptor",
#                 event="DecryptAttempt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "encrypt_log_not_found",
#                     "pop": pop_hex,
#                     "metadata": metadata
#                 }
#             )
#             raise ValueError("Matching encryption log entry not found for provided PoP")

#         hctx = encrypt_log["details"].get("hctx")
#         if not hctx:
#             self.logger.log(
#                 component="Decryptor",
#                 event="DecryptAttempt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "hctx_missing_in_log",
#                     "pop": pop_hex
#                 }
#             )
#             raise ValueError("hctx not present in encryption log entry")
        
#         # Recompute salt from role+policy+epoch
#         policy_text = self.policy_store.resolve(role, epoch)
#         salt = _hash(role + policy_text + epoch)
        
#         # 2) Resolve policy
#         try:
#             policy_text = self.policy_store.resolve(role, epoch)
#         except Exception as e:
#             self.logger.log(
#                 component="Decryptor",
#                 event="DecryptAttempt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "policy_resolve_failed",
#                     "role": role,
#                     "epoch": epoch,
#                     "error": str(e)
#                 }
#             )
#             raise

#         # 3) Derive Fernet key
#         fernet_key = _fernet_key_from_material(hctx, salt)
#         cipher = Fernet(fernet_key)

#         # 4) Decrypt ciphertext
#         ciphertext = encrypted_packet["ciphertext"]
#         try:
#             plaintext_bytes = cipher.decrypt(ciphertext.encode())
#         except InvalidToken:
#             self.logger.log(
#                 component="Decryptor",
#                 event="DecryptAttempt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "decrypt_failed",
#                     "pop": pop_hex,
#                     "role": role,
#                     "epoch": epoch
#                 }
#             )
#             raise ValueError("Decryption failed (invalid token / wrong key)")

#         # 5) Load JSON
#         try:
#             recovered_payload = json.loads(plaintext_bytes.decode())
#         except Exception:
#             self.logger.log(
#                 component="Decryptor",
#                 event="DecryptAttempt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "invalid_plaintext_json",
#                     "pop": pop_hex
#                 }
#             )
#             raise ValueError("Plaintext is not valid JSON")

#         # 6) Verify PoP (canonicalized JSON + role + epoch)
#         recomputed_pop = _hash(
#             json.dumps(recovered_payload, sort_keys=True, separators=(",", ":")) + role + epoch
#         )
#         if recomputed_pop == pop_hex:
#             self.logger.log(
#                 component="Decryptor",
#                 event="Decrypt",
#                 details={
#                     "status": "SUCCESS",
#                     "pop": pop_hex,
#                     "role": role,
#                     "epoch": epoch,
#                     "hctx": hctx
#                 }
#             )
#             return recovered_payload
#         else:
#             self.logger.log(
#                 component="Decryptor",
#                 event="Decrypt",
#                 details={
#                     "status": "FAIL",
#                     "reason": "pop_mismatch",
#                     "expected_pop": recomputed_pop,
#                     "observed_pop": pop_hex,
#                     "role": role,
#                     "epoch": epoch,
#                     "hctx": hctx
#                 }
#             )
#             raise ValueError("PoP verification failed (integrity check)")


# # ------------------------------
# # Demo
# # ------------------------------
# # if __name__ == "__main__":
# #     from prompter import Prompter
# #     from cryptor import Cryptor
# #     from dotenv import load_dotenv
# #     import os
# #     load_dotenv()

# #     api_key = os.getenv("GEMINI_API_KEY")
# #     logger = AuditLogger()
# #     ps = PolicyStore(policies={"TellerAgent": "≤10000"})

# #     p = Prompter(api_key=api_key, logger=logger)
# #     c = Cryptor(logger=logger)

# #     instruction = "transfer $9000 from 56789 to 45678"
# #     normalized = p.normalize_instruction(instruction)
# #     packet = c.encrypt(normalized)

# #     d = Decryptor(logger=logger, policy_store=ps)
# #     recovered = d.decrypt(packet, expected_role="TellerAgent")

# #     print("Recovered payload:", recovered)
# #     print("\nAudit log chain:")
# #     for e in logger.logs:
# #         print(e)

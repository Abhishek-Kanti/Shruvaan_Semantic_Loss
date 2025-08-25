import hashlib
import json
import base64
from typing import Any, Dict, Union, Optional, List
from cryptography.fernet import Fernet
from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger

JsonObj = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class Cryptor:
    """
    Per-field JSON crypter:
    - Derives a Fernet key via HKP (hctx + salt)
    - Encrypts every key and every leaf value (type preserved via JSON-encode)
    - Preserves JSON structure (dict/list)
    - Emits enc_map + PoP + metadata
    """

    def __init__(self, logger: Optional[AuditLogger] = None,
                 include_kdf_hints: bool = True,
                 history_logger: Optional[CryptoHistoryLogger] = None):
        self.logger = logger or AuditLogger()
        self.include_kdf_hints = include_kdf_hints
        self.history_logger = history_logger   # shared logger injected here

    # ----- HKP / helpers -----------------------------------------------------

    def _hash_hex(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str) -> bytes:
        material = (hctx_hex + salt_hex).encode("utf-8")
        return hashlib.sha256(material).digest()  # 32 bytes

    def _to_canonical(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _enc_str(self, cipher: Fernet, s: str) -> str:
        return cipher.encrypt(s.encode("utf-8")).decode("utf-8")

    # ----- structure encryption ---------------------------------------------

    def _encrypt_structure(self, cipher: Fernet, obj: JsonObj) -> JsonObj:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    k = json.dumps(k, separators=(",", ":"))
                ek = self._enc_str(cipher, k)
                out[ek] = self._encrypt_structure(cipher, v)
            return out

        if isinstance(obj, list):
            return [self._encrypt_structure(cipher, v) for v in obj]

        # scalar leaf -> JSON-encode (preserves type) before encrypt
        encoded = json.dumps(obj, separators=(",", ":"))
        return self._enc_str(cipher, encoded)

    # ----- public API --------------------------------------------------------

    def encrypt(self, normalized_instruction: Dict[str, Any]) -> Dict[str, Any]:
        raw_instruction = normalized_instruction.get("raw_instruction", "")
        role = normalized_instruction.get("role", "")
        policy = normalized_instruction.get("policy", "")
        epoch = normalized_instruction.get("epoch", "")

        # 1. KDF material
        hctx_hex = self._hash_hex(raw_instruction)
        salt_hex = self._hash_hex(role + policy + epoch)

        # 2. Fernet key
        hkp_key_bytes = self._hkp_key_bytes(hctx_hex, salt_hex)
        fernet_key = base64.urlsafe_b64encode(hkp_key_bytes)
        cipher = Fernet(fernet_key)

        # 3. Canonical plaintext
        canonical_plain = self._to_canonical(normalized_instruction)

        # 4. Encrypt structure
        enc_map = self._encrypt_structure(cipher, normalized_instruction)

        # 5. Proof of possession
        pop = self._hash_hex(canonical_plain + role + epoch)

        # 6. Metadata
        metadata = {"role": role, "epoch": epoch}
        if self.include_kdf_hints:
            metadata["kdf"] = {"hctx": hctx_hex, "salt": salt_hex}

        packet = {"enc_map": enc_map, "pop": pop, "metadata": metadata}

        # 7. Log encrypted side for history (no plaintext yet)
        self.history_logger.log_pair(packet, None)

        # 8. Audit
        self.logger.log(
            component="Cryptor",
            event="Encrypt",
            details={
                "hctx": hctx_hex if self.include_kdf_hints else "<hidden>",
                "salt": salt_hex if self.include_kdf_hints else "<hidden>",
                "role": role,
                "epoch": epoch,
                "shape": f"dict:{len(normalized_instruction)}",
            },
        )

        return packet



# import hashlib
# import json
# import base64
# from cryptography.fernet import Fernet
# from audit_logger import AuditLogger

# class Cryptor:
#     def __init__(self, logger: AuditLogger = None):
#         self.logger = logger or AuditLogger()

#     def _hash_hex(self, s: str) -> str:
#         return hashlib.sha256(s.encode()).hexdigest()

#     def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str) -> bytes:
#         # Derive HKP key as raw 32-byte digest from concatenated hex strings.
#         material = (hctx_hex + salt_hex).encode()
#         return hashlib.sha256(material).digest()  # 32 bytes

#     def encrypt(self, normalized_instruction: dict):
#         # 1) hctx from raw instruction
#         hctx_hex = self._hash_hex(normalized_instruction["raw_instruction"])

#         # 2) salt = H(role + policy + epoch)  (hex)
#         role = normalized_instruction.get("role", "")
#         policy = normalized_instruction.get("policy", "")
#         epoch = normalized_instruction.get("epoch", "")
#         salt_hex = self._hash_hex(role + policy + epoch)

#         # 3) HKP key bytes & Fernet key
#         hkp_key_bytes = self._hkp_key_bytes(hctx_hex, salt_hex)
#         fernet_key = base64.urlsafe_b64encode(hkp_key_bytes)  # 32-byte -> urlsafe b64

#         cipher = Fernet(fernet_key)

#         # 4) Canonicalize payload (must match decryptor)
#         canonical_payload = json.dumps(
#             normalized_instruction, sort_keys=True, separators=(",", ":")
#         )

#         ciphertext = cipher.encrypt(canonical_payload.encode()).decode()

#         # 5) PoP over canonical payload + role + epoch
#         pop = self._hash_hex(canonical_payload + role + epoch)

#         # 6) Packet (do NOT include salt)
#         encrypted_packet = {
#             "ciphertext": ciphertext,
#             "pop": pop,
#             "metadata": {
#                 "role": role,
#                 "epoch": epoch
#             }
#         }

#         # 7) Audit log (salt is derivable but logging it is OK)
#         self.logger.log(
#             component="Cryptor",
#             event="Encrypt",
#             details={
#                 "hctx": hctx_hex,
#                 "pop": pop,
#                 "role": role,
#                 "epoch": epoch
#             }
#         )

#         return encrypted_packet


# # --- Quick test ---
# # if __name__ == "__main__":
# #     from prompter import Prompter
# #     from dotenv import load_dotenv
# #     import os

# #     load_dotenv()
# #     api_key = os.getenv("GEMINI_API_KEY")

# #     p = Prompter(api_key=api_key)
# #     c = Cryptor()

# #     instruction = "transfer $9000 from 56789 to 45678"
# #     norm = p.normalize_instruction(instruction)
# #     encrypted = c.encrypt(norm)

# #     print("\nNormalized Instruction:\n", norm)
# #     print("\nEncrypted Packet:\n", encrypted)

# #     # Only show log chain once
# #     print("\nAudit Log Chain:")
# #     for entry in c.logger.logs:
# #         print(entry)

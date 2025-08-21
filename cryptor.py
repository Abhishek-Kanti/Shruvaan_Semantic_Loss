import hashlib
import json
import base64
from cryptography.fernet import Fernet
from audit_logger import AuditLogger

class Cryptor:
    def __init__(self, logger: AuditLogger = None):
        self.logger = logger or AuditLogger()

    def _hash_hex(self, s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str) -> bytes:
        # Derive HKP key as raw 32-byte digest from concatenated hex strings.
        material = (hctx_hex + salt_hex).encode()
        return hashlib.sha256(material).digest()  # 32 bytes

    def encrypt(self, normalized_instruction: dict):
        # 1) hctx from raw instruction
        hctx_hex = self._hash_hex(normalized_instruction["raw_instruction"])

        # 2) salt = H(role + policy + epoch)  (hex)
        role = normalized_instruction.get("role", "")
        policy = normalized_instruction.get("policy", "")
        epoch = normalized_instruction.get("epoch", "")
        salt_hex = self._hash_hex(role + policy + epoch)

        # 3) HKP key bytes & Fernet key
        hkp_key_bytes = self._hkp_key_bytes(hctx_hex, salt_hex)
        fernet_key = base64.urlsafe_b64encode(hkp_key_bytes)  # 32-byte -> urlsafe b64

        cipher = Fernet(fernet_key)

        # 4) Canonicalize payload (must match decryptor)
        canonical_payload = json.dumps(
            normalized_instruction, sort_keys=True, separators=(",", ":")
        )

        ciphertext = cipher.encrypt(canonical_payload.encode()).decode()

        # 5) PoP over canonical payload + role + epoch
        pop = self._hash_hex(canonical_payload + role + epoch)

        # 6) Packet (do NOT include salt)
        encrypted_packet = {
            "ciphertext": ciphertext,
            "pop": pop,
            "metadata": {
                "role": role,
                "epoch": epoch
            }
        }

        # 7) Audit log (salt is derivable but logging it is OK)
        self.logger.log(
            component="Cryptor",
            event="Encrypt",
            details={
                "hctx": hctx_hex,
                "pop": pop,
                "role": role,
                "epoch": epoch
            }
        )

        return encrypted_packet


# --- Quick test ---
# if __name__ == "__main__":
#     from prompter import Prompter
#     from dotenv import load_dotenv
#     import os

#     load_dotenv()
#     api_key = os.getenv("GEMINI_API_KEY")

#     p = Prompter(api_key=api_key)
#     c = Cryptor()

#     instruction = "transfer $9000 from 56789 to 45678"
#     norm = p.normalize_instruction(instruction)
#     encrypted = c.encrypt(norm)

#     print("\nNormalized Instruction:\n", norm)
#     print("\nEncrypted Packet:\n", encrypted)

#     # Only show log chain once
#     print("\nAudit Log Chain:")
#     for entry in c.logger.logs:
#         print(entry)

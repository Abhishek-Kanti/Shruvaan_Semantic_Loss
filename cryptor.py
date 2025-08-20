import hashlib
import json
import base64
from cryptography.fernet import Fernet
from datetime import datetime, timezone
from audit_logger import AuditLogger


class Cryptor:
    def __init__(self, logger: AuditLogger = None):
        self.logger = logger or AuditLogger()
        
    def _hash(self, data: str):
        return hashlib.sha256(data.encode()).hexdigest()

    def encrypt(self, normalized_instruction: dict):
        # Step 1: Semantic hash of raw instruction
        hctx = self._hash(normalized_instruction["raw_instruction"])

        # Step 2: HKP salt based on role + policy + epoch
        role = normalized_instruction.get("role", "")
        policy = normalized_instruction.get("policy", "")
        epoch = normalized_instruction.get("epoch", "")
        salt = self._hash(role + policy + epoch)

        # Step 3: HKP key = H(hctx + salt)
        hkp_key = self._hash(hctx + salt)

        # Step 4: Symmetric encryption key (Fernet requires 32-byte base64)
        fernet_key = base64.urlsafe_b64encode(hkp_key.encode()[:32])
        cipher = Fernet(fernet_key)

        payload = json.dumps(normalized_instruction).encode()
        ciphertext = cipher.encrypt(payload).decode()

        # Step 5: Proof-of-Protocol (PoP) = H(payload + role + epoch)
        pop = self._hash(json.dumps(normalized_instruction) + role + epoch)

        # Step 6: Bundle result
        encrypted_packet = {
            "ciphertext": ciphertext,
            "hkp_salt": salt,
            "pop": pop,
            "metadata": {
                "role": role,
                "epoch": epoch
            }
        }

        # Step 7: Audit log
        self.logger.log(
            component="Cryptor",
            event="Encrypt",
            details={
                "hctx": hctx,
                "hkp_salt": salt,
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

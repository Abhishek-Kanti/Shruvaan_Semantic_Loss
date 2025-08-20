from audit_logger import AuditLogger
from prompter import Prompter
from cryptor import Cryptor
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# One logger instance for the whole pipeline
logger = AuditLogger()

# Pass same logger everywhere
p = Prompter(api_key=api_key, logger=logger)
c = Cryptor(logger=logger)

instruction = "transfer $9000 from 56789 to 45678"

# Normalize → Encrypt
norm = p.normalize_instruction(instruction)
encrypted = c.encrypt(norm)

print("\nNormalized Instruction:\n", norm)
print("\nEncrypted Packet:\n", encrypted)

# Export once
logger.export("audit_log.json")
print("Unified audit log exported → audit_log.json")

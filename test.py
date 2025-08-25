from audit_logger import AuditLogger
from prompter import Prompter
from cryptor import Cryptor
from decryptor import Decryptor
from mimicus import run_mimicus
import os
from dotenv import load_dotenv
import json
from crypto_history_logger import CryptoHistoryLogger

# === Setup ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

logger = AuditLogger()
history_logger = CryptoHistoryLogger()

p = Prompter(api_key=api_key, logger=logger)
c = Cryptor(logger=logger, history_logger=history_logger)
d = Decryptor(logger=logger, history_logger=history_logger)

print("\n=== Instruction ===")
instruction = "transfer $1000 from 42000 to 60000"
print(f"Instruction: {instruction}")

# 1. Normalize
print("\n=== Normalize ===")
normalized = p.normalize_instruction(instruction)
print(json.dumps(normalized, indent=2))

# 2. Encrypt
print("\n=== Encrypt ===")
packet = c.encrypt(normalized)
print(json.dumps(packet, indent=2)[:500] + " ...")  # trunc to avoid huge output

# 3. Decrypt
print("\n=== Decrypt ===")
recovered = d.decrypt(packet)
print(json.dumps(recovered, indent=2))

# 4. Run Mimicus probe on decrypted output
print("\n=== Running Mimicus Probe ===")
mimic_result = run_mimicus(recovered, packet, logger=logger, history_logger=history_logger)

print("\nMimic Fields:")
print(json.dumps(mimic_result["mimic_fields"], indent=2))

print("\nLeakage Components:")
print(json.dumps(mimic_result["components"], indent=2))

print(f"\nLeakage Score: {mimic_result['leakage_score']:.3f}")
print(f"Decision: {mimic_result['decision']}")

# === Export unified log ===
print("\n=== Export unified log ===")
logger.export("audit_log.json")
print("Unified audit log exported → audit_log.json")

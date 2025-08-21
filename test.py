from audit_logger import AuditLogger
from prompter import Prompter
from cryptor import Cryptor
from decryptor import Decryptor
from mimicus import run_mimicus
from policy import PolicyStore
import os
from dotenv import load_dotenv
import json

# === Setup ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Shared audit logger (single log file for all components)
logger = AuditLogger()
ps = PolicyStore(policies={"TellerAgent": "≤10000"})

p = Prompter(api_key=api_key, logger=logger)
c = Cryptor(logger=logger)
d = Decryptor(logger=logger, policy_store=ps)

print("\n=== Instruction ===")
instruction = "transfer $9000 from 56789 to 45678"
print(f"Instruction: {instruction}")

# 1. Normalize
print("\n=== Normalize ===")
normalized = p.normalize_instruction(instruction)
print(json.dumps(normalized, indent=2))

# 2. Encrypt
print("\n=== Encrypt ===")
packet = c.encrypt(normalized)
print(json.dumps(packet, indent=2))

# 3. Decrypt
print("\n=== Decrypt ===")
recovered = d.decrypt(packet, expected_role="TellerAgent")
print(json.dumps(recovered, indent=2))

# 4. Run Mimicus probe on decrypted output
print("\n=== Running Mimicus Probe ===")
mimic_result = run_mimicus(recovered)

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

# from audit_logger import AuditLogger
# from prompter import Prompter
# from cryptor import Cryptor
# from decryptor import Decryptor
# from mimicus import run_mimicus
# from probator import run_probator
# from praeceptor import Praeceptor   
# import os
# from dotenv import load_dotenv
# import json
# from crypto_history_logger import CryptoHistoryLogger

# # === Setup ===
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# logger = AuditLogger()
# history_logger = CryptoHistoryLogger()

# p = Prompter(api_key=api_key, logger=logger)
# c = Cryptor(logger=logger, history_logger=history_logger)
# d = Decryptor(logger=logger, history_logger=history_logger)
# praeceptor = Praeceptor(logger=logger, history_logger=history_logger)   # instance

# print("\n=== Instruction ===")
# instruction = "transfer $1500 from 6395-8845-2791 to 6559-6423-4401"
# print(f"Instruction: {instruction}")

# # 1. Normalize
# print("\n=== Normalize ===")
# normalized = p.normalize_instruction(instruction)
# print(json.dumps(normalized, indent=2))

# # 2. Encrypt (initial θ = None)
# print("\n=== Encrypt (Initial) ===")
# packet = c.encrypt(normalized)
# print(json.dumps(packet, indent=2)[:500] + " ...")

# # 3. Decrypt
# print("\n=== Decrypt ===")
# recovered = d.decrypt(packet)
# print(json.dumps(recovered, indent=2))

# # 4. Run Praeceptor update step
# # praeceptor internally runs Mimicus and Probator as well
# print("\n=== Praeceptor Step Update ===")
# pre_result = praeceptor.step_update(recovered, packet)
# theta_new = pre_result["theta"]
# c.set_theta(theta_new)   # update Cryptor with new θ
# print(f"New θ set by Praeceptor: {theta_new}")
# print("Leakage metrics:", pre_result["leakage"])

# # 5. Encrypt again with updated θ
# print("\n=== Re-Encrypt with Updated θ ===")
# packet2 = c.encrypt(normalized)
# print(json.dumps(packet2, indent=2)[:500] + " ...")

# # 6. Decrypt again
# print("\n=== Re-Decrypt (Backward Compatible) ===")
# recovered2 = d.decrypt(packet2)
# print(json.dumps(recovered2, indent=2))

# # 7. Probe again after Praeceptor update
# print("\n=== Mimicus Probe (Post-Update) ===")
# mimic_result2 = run_mimicus(recovered2, packet2, logger=logger, history_logger=history_logger)
# print(json.dumps(mimic_result2, indent=2))

# print("\n=== Probator Probe (Post-Update) ===")
# probator_result2 = run_probator(recovered2, packet2, logger=logger)
# print(json.dumps(probator_result2, indent=2))

# # === Export unified log ===
# print("\n=== Export unified log ===")
# logger.export("audit_log.json")
# print("Unified audit log exported → audit_log.json")

from audit_logger import AuditLogger
from prompter import Prompter
from cryptor import Cryptor
from decryptor import Decryptor
from mimicus import run_mimicus
from probator import run_probator
from praeceptor import Praeceptor   
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
praeceptor = Praeceptor(logger=logger, history_logger=history_logger)   # instance

print("\n=== Instruction ===")
instruction = "transfer $1500 from 6395-8845-2791 to 6559-6423-4401"
print(f"Instruction: {instruction}")

# 1. Normalize
print("\n=== Normalize ===")
normalized = p.normalize_instruction(instruction)
print(json.dumps(normalized, indent=2))

# 2. Encrypt (initial θ = None)
print("\n=== Encrypt (Initial) ===")
packet = c.encrypt(normalized)
print(json.dumps(packet, indent=2)[:500] + " ...")

# 3. Decrypt
print("\n=== Decrypt ===")
recovered = d.decrypt(packet)
print(json.dumps(recovered, indent=2))

# 4. Run Praeceptor training loop until safe or max steps
print("\n=== Praeceptor Training (Inner Loop) ===")
praeceptor = Praeceptor(logger=logger, history_logger=history_logger, cryptor=c, decryptor=d)
result = praeceptor.train_until_safe(normalized, safe_threshold=0.25, max_steps=50)

print("\n=== Praeceptor Training Result ===")
print("Success:", result["success"])
print("Final θ:", result["final_theta"])
# if result["history"]:
#     last = result["history"][-1]
#     print(f"Final Leakage -> Mimic: {last['mimic']:.4f}, Prob: {last['prob']:.4f}, Combined: {last['combined']:.4f}")
#     print(f"Total Steps Run: {len(result['history'])}")

# 5. Encrypt again with updated θ
print("\n=== Re-Encrypt with Updated θ ===")
packet2 = c.encrypt(normalized)
print(json.dumps(packet2, indent=2)[:500] + " ...")

# 6. Decrypt again
print("\n=== Re-Decrypt (Backward Compatible) ===")
recovered2 = d.decrypt(packet2)
print(json.dumps(recovered2, indent=2))

# 7. Probe again after Praeceptor update
print("\n=== Mimicus Probe (Post-Update) ===")
mimic_result2 = run_mimicus(recovered2, packet2, logger=logger, history_logger=history_logger)
print(json.dumps(mimic_result2, indent=2))

print("\n=== Probator Probe (Post-Update) ===")
probator_result2 = run_probator(recovered2, packet2, logger=logger)
print(json.dumps(probator_result2, indent=2))

# === Export unified log ===
print("\n=== Export unified log ===")
logger.export("audit_log.json")
print("Unified audit log exported → audit_log.json")

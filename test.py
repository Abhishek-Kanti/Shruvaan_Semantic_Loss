from audit_logger import AuditLogger
from prompter import Prompter
from cryptor import Cryptor
from decryptor import Decryptor
from praeceptor import Praeceptor   
import os
from dotenv import load_dotenv
import json
from crypto_history_logger import CryptoHistoryLogger

# === Setup ===
load_dotenv()
api_key_gemi = os.getenv("GEMINI_API_KEY")
api_key_groq = os.getenv("GROQ_API_KEY")

provider_gemi = "gemini"   # or "openai", "groq"
provider_groq = "groq"    # second provider for Praeceptor

logger = AuditLogger(filepath="audit_log.json")
history_logger = CryptoHistoryLogger()

p = Prompter(provider=provider_gemi, api_key=api_key_gemi, logger=logger)
c = Cryptor(logger=logger, history_logger=history_logger)
d = Decryptor(logger=logger, history_logger=history_logger)
praeceptor = Praeceptor(
    provider=provider_groq,
    api_key=api_key_groq,
    logger=logger,
    history_logger=history_logger,
    cryptor=c,
    decryptor=d
)

print("\n=== Instruction ===")
instruction = "transfer $1500 from 6495-8845-2799 to 9459-6423-4401"
print(f"Instruction: {instruction}")

# 1. Normalize
print("\n=== Normalize ===")
normalized = p.normalize_instruction(instruction)
print(json.dumps(normalized, indent=2))

# 2. Run Praeceptor training loop until safe or max steps
print("\n=== Praeceptor Training (Inner Loop) ===")
result = praeceptor.train_until_safe(normalized, safe_threshold=0.25, max_steps=5, verbose=True)

print("\n=== Praeceptor Training Result ===")
print("Success:", result["success"])
print("Final θ:", result["final_theta"])

# Print θ evolution during training
print("\n=== θ Evolution During Training ===")
for step_hist in result["history"]:
    print(f"Step {step_hist['step']}: θ={step_hist['theta']} mimic={step_hist['mimic']:.4f} prob={step_hist['prob']:.4f}")

# 3. Encrypt (with whatever θ is now in cryptor)
print("\n=== Encrypt (Outside Praeceptor) ===")
packet = c.encrypt(normalized)
print(json.dumps(packet, indent=2)[:500] + " ...")

# 4. Decrypt
print("\n=== Decrypt (Outside Praeceptor) ===")
recovered = d.decrypt(packet)
print(json.dumps(recovered, indent=2))

# === Export unified log ===
print("\n=== Export unified log ===")
print("Unified audit log exported → audit_log.json")
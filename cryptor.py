import hashlib
import json
import torch
import string
import base64
from typing import Any, Dict, Union, Optional, List
from cryptography.fernet import Fernet
from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger

JsonObj = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class Cryptor:
    """
    Per-field JSON crypter with adaptive theta support:
    - Derives a Fernet key via HKP (hctx + salt, optionally modulated by theta)
    - Encrypts every key and every leaf value (type preserved via JSON-encode)
    - Preserves JSON structure (dict/list)
    - Emits enc_map + PoP + metadata
    """

    def __init__(self, logger: Optional[AuditLogger] = None,
                 include_kdf_hints: bool = True,
                 history_logger: Optional[CryptoHistoryLogger] = None,
                 theta: Optional[List[float]] = None):
        self.logger = logger or AuditLogger()
        self.include_kdf_hints = include_kdf_hints
        self.history_logger = history_logger   # shared logger injected here

        # θ parameters (adaptive encryption knobs)
        self.theta = theta or None

    # ----- Theta management --------------------------------------------------

    def set_theta(self, theta: List[float]):
        """Set encryption control parameters θ (from Preceptor)."""
        self.theta = theta

    def get_theta(self) -> Optional[List[float]]:
        """Return current θ vector (if any)."""
        return self.theta

    # ----- HKP / helpers -----------------------------------------------------

    def _hash_hex(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _apply_theta(self, hctx_hex: str, salt_hex: str) -> str:
        """
        Mix theta into the HKP material. Keeps backward compatibility if theta=None.
        """
        if self.theta is None:
            return hctx_hex + salt_hex

        # Simple modulation: mix theta values into string
        theta_str = ",".join([f"{t:.4f}" for t in self.theta])
        return hctx_hex + salt_hex + theta_str

    def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str) -> bytes:
        material = self._apply_theta(hctx_hex, salt_hex).encode("utf-8")
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

    # def encrypt(self, normalized_instruction: Dict[str, Any], logging = True) -> Dict[str, Any]:
    #     raw_instruction = normalized_instruction.get("raw_instruction", "")
    #     role = normalized_instruction.get("role", "")
    #     policy = normalized_instruction.get("policy", "")
    #     epoch = normalized_instruction.get("epoch", "")

    #     # 1. KDF material
    #     hctx_hex = self._hash_hex(raw_instruction)
    #     salt_hex = self._hash_hex(role + policy + epoch)

    #     # 2. Fernet key (now possibly modulated by θ)
    #     hkp_key_bytes = self._hkp_key_bytes(hctx_hex, salt_hex)
    #     fernet_key = base64.urlsafe_b64encode(hkp_key_bytes)
    #     cipher = Fernet(fernet_key)

    #     # 3. Canonical plaintext
    #     canonical_plain = self._to_canonical(normalized_instruction)

    #     # 4. Encrypt structure
    #     enc_map = self._encrypt_structure(cipher, normalized_instruction)

    #     # 5. Proof of possession
    #     pop = self._hash_hex(canonical_plain + role + epoch)

    #     # 6. Metadata
    #     metadata = {"role": role, "epoch": epoch}
    #     if self.include_kdf_hints:
    #         metadata["kdf"] = {"hctx": hctx_hex, "salt": salt_hex}
    #     if self.theta is not None:
    #         metadata["theta"] = self.theta  # track applied theta for audit

    #     packet = {"enc_map": enc_map, "pop": pop, "metadata": metadata}

    #     # 7. Audit
    #     if(logging == True):
    #         self.logger.log(
    #             component="Cryptor",
    #             event="Encrypt",
    #             details={
    #                 "hctx": hctx_hex if self.include_kdf_hints else "<hidden>",
    #                 "salt": salt_hex if self.include_kdf_hints else "<hidden>",
    #                 "role": role,
    #                 "epoch": epoch,
    #                 "theta": self.theta if self.theta else "<default>",
    #                 "shape": f"dict:{len(normalized_instruction)}",
    #             },
    #         )

    #     return packet

    def encrypt(self, normalized_instruction: Dict[str, Any], logging: bool = True) -> Dict[str, Any]:
        """
        Example encryption pipeline where θ influences obfuscation parameters.
        θ values are in [0,1] and map onto PARAMS = [
            'obfuscation_depth', 'noise_scale', 'mask_rate', 'unicode_rate', 'length_jitter'
        ]
        """
        packet = {"enc_map": {}}

        obfuscation_depth = int(1 + self.theta[0] * 3)   # 1..4 levels
        noise_scale       = self.theta[1]                # 0..1 fraction of chars noised
        mask_rate         = self.theta[2]                # 0..1 fraction masked
        unicode_rate      = self.theta[3]                # 0..1 chance of unicode substitution
        length_jitter     = int(self.theta[4] * 5)       # add/remove up to ±5 chars

        for k, v in normalized_instruction.items():
            if isinstance(v, str):
                text = v
                # apply obfuscation depth
                for _ in range(obfuscation_depth):
                    text = text[::-1]  # naive: reverse repeatedly (placeholder)

            # apply noise
            if noise_scale > 0:
                chars = list(text)
                for i in range(len(chars)):
                    if torch.rand(1).item() < noise_scale * 0.1:  # mild noise
                        chars[i] = string.ascii_uppercase[torch.randint(0, len(string.ascii_uppercase), (1,)).item()]
                text = "".join(chars)

            # apply mask
            if mask_rate > 0:
                chars = list(text)
                for i in range(len(chars)):
                    if torch.rand(1).item() < mask_rate:
                        chars[i] = "*"
                text = "".join(chars)

            # apply unicode substitution
            if unicode_rate > 0:
                chars = list(text)
                for i in range(len(chars)):
                    if torch.rand(1).item() < unicode_rate * 0.05:
                        chars[i] = chr(0x2500 + (ord(chars[i]) % 256))  # box-drawing substitute
                text = "".join(chars)

            # length jitter
            if length_jitter > 0:
                jitter = torch.randint(-length_jitter, length_jitter + 1, (1,)).item()
                if jitter > 0:
                    text = text + ("#" * jitter)
                elif jitter < 0:
                    text = text[:jitter]

                packet["enc_map"][k] = text
            else:
                packet["enc_map"][k] = v

        if logging and self.logger:
            try:
                self.logger.log("Cryptor", "Encrypt", {"theta": self.theta})
            except Exception:
                pass

        return packet
# cryptor.py
import hashlib
import json
import os
import base64
from typing import Any, Dict, Union, Optional, List

from cryptography.fernet import Fernet
from utils.audit_logger import AuditLogger
from utils.crypto_history_logger import CryptoHistoryLogger

import torch
import warnings

JsonObj = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class Cryptor:
    """
    Per-field JSON crypter with adaptive theta support (refactored):
    - derive Fernet key via HKP (hctx + salt), optional modulation by theta (passed in)
    - encrypts every key and every leaf value (type preserved via JSON-encode)
    - preserves JSON structure (dict/list)
    - emits enc_map + PoP + metadata (kdf hints)
    """

    def __init__(self, logger: Optional[AuditLogger] = None,
                 include_kdf_hints: bool = True,
                 history_logger: Optional[CryptoHistoryLogger] = None,
                 theta: Optional[List[float]] = None):
        self.logger = logger or AuditLogger()
        self.include_kdf_hints = include_kdf_hints
        self.history_logger = history_logger   # shared logger injected here

        # θ parameters (adaptive encryption knobs)
        # default to mid-range values if not supplied
        # Keep as a simple python list (not a tensor). Praeceptor will pass its theta explicitly when training.
        self.theta = theta or [0.5, 0.5, 0.5, 0.5, 0.5]
        self._checkpoint_theta_cache = None  # cache last-loaded theta from checkpoint

    # ----- Theta management --------------------------------------------------

    def set_theta(self, theta: List[float]):
        """Set encryption control parameters θ (from Praeceptor or caller)."""
        self.theta = theta

    def get_theta(self) -> Optional[List[float]]:
        """Return current θ vector (if any)."""
        return self.theta

    def _load_theta_from_checkpoint(self, filepath: str) -> Optional[List[float]]:
        """
        Load θ from Praeceptor checkpoint file (if any).
        Returns None if file missing or invalid.
        """
        if self._checkpoint_theta_cache is not None:
            return self._checkpoint_theta_cache  # use cached value
        try:
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
            ckpt = torch.load(filepath, map_location="cpu", weights_only=True)
            theta_tensor = ckpt["cryptor"]["theta"]
            theta = torch.sigmoid(theta_tensor).detach().cpu().numpy().tolist()
            self._checkpoint_theta_cache = self._validate_theta(theta)  # cache it
            return theta
        except Exception:
            return None

    # ----- HKP / helpers -----------------------------------------------------

    def _hash_hex(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _validate_theta(self, theta):
        if not theta or not isinstance(theta, (list, tuple)):
            return None
        out = []
        for t in theta:
            try:
                out.append(float(t))
            except Exception:
                continue
        return out if out else None
    
    def _apply_theta_to_material(self, hctx_hex: str, salt_hex: str, theta: Optional[List[float]]) -> str:
        """
        Combine hctx + salt + theta into a single string used for key derivation.
        If theta is None or invalid, fallback to hctx + salt (backward compatible).
        """
        if theta is None:
            return hctx_hex + salt_hex

        # Validate & sanitize theta: keep numeric values only
        theta_vals = []
        for t in theta:
            try:
                theta_vals.append(float(t))
            except Exception:
                continue
        if not theta_vals:
            return hctx_hex + salt_hex

        theta_str = ",".join([f"{float(t):.4f}" for t in theta_vals])
        return hctx_hex + salt_hex + theta_str

    def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str, theta: Optional[List[float]]) -> bytes:
        material = self._apply_theta_to_material(hctx_hex, salt_hex, theta).encode("utf-8")
        return hashlib.sha256(material).digest()  # 32 bytes

    def _to_canonical(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _enc_str(self, cipher: Fernet, s: str) -> str:
        return cipher.encrypt(s.encode("utf-8")).decode("utf-8")

    # ----- structure encryption ---------------------------------------------

    def _encrypt_structure(self, cipher: Fernet, obj: JsonObj) -> JsonObj:
        """
        Recursively encrypt dict keys and values and list elements.
        Scalar leaves are JSON-encoded (to preserve types) then encrypted.
        """
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                # ensure key is a string canonical form
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

    # --------------------- public API ----------------------------------------------

    def encrypt(self, normalized_instruction: Dict[str, Any], 
                theta: Optional[List[float]] = None,
                use_checkpoint=True, 
                logging: bool = True) -> Dict[str, Any]:
        """
        Proper encryption pipeline:
        - create random hctx & salt (hex)
        - derive key (HKP + theta) -> Fernet
        - encrypt structure
        - compute PoP and attach metadata.kdf {hctx,salt} (if configured)
        - log (hctx, salt, theta) to audit logger for later retrieval by Decryptor
        """
        if theta is not None:
            use_theta = self._validate_theta(theta)
        elif use_checkpoint and os.path.exists("praeceptor_checkpoint.pt"):
            use_theta = self._validate_theta(self._load_theta_from_checkpoint("praeceptor_checkpoint.pt"))
        elif self.theta is not None:
            use_theta = self._validate_theta(self.theta)
        else:
            use_theta = [0.5] * 5   # default

        # 1) KDF hints (random per-packet)
        hctx = os.urandom(16)
        salt = os.urandom(16)
        hctx_hex = hctx.hex()
        salt_hex = salt.hex()

        # 2) derive fernet key using theta-mixed HKP
        key_bytes = self._hkp_key_bytes(hctx_hex, salt_hex, use_theta)  # 32 bytes
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        cipher = Fernet(fernet_key)

        # 3) encrypt structure (per-key & per-value)
        enc_map = self._encrypt_structure(cipher, normalized_instruction)

        # 4) compute PoP = sha256(canonical_plain + role + epoch)
        role = normalized_instruction.get("role", "") if isinstance(normalized_instruction, dict) else ""
        epoch = normalized_instruction.get("epoch", "") if isinstance(normalized_instruction, dict) else ""
        canonical_plain = self._to_canonical(normalized_instruction)
        pop = self._hash_hex(canonical_plain + role + epoch)

        # 5) assemble packet
        packet: Dict[str, Any] = {"enc_map": enc_map, "pop": pop, "metadata": {}}
        if self.include_kdf_hints:
            packet["metadata"]["kdf"] = {"hctx": hctx_hex, "salt": salt_hex}
        packet["metadata"]["role"] = role
        packet["metadata"]["epoch"] = epoch

        # 6) audit log: save hctx/salt/theta so Decryptor.get_theta_for_packet can find theta later
        if logging and self.logger:
            try:
                # Store the theta that was actually used for this packet (explicit or internal)
                self.logger.log("Cryptor", "Encrypt", {
                    "hctx": hctx_hex,
                    "salt": salt_hex,
                    "theta": use_theta
                })
            except Exception:
                # never fail encryption because logging misbehaved
                pass
        
        self.theta = use_theta  # update internal theta to the one used
        return packet

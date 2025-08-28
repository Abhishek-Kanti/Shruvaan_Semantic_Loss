import hashlib
import json
import base64
from typing import Any, Dict, Union, Optional, List
from cryptography.fernet import Fernet, InvalidToken
from audit_logger import AuditLogger
from crypto_history_logger import CryptoHistoryLogger

JsonObj = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class Decryptor:
    """
    Inverse of Cryptor:
    - Rebuild Fernet from HKP
    - Decrypt every key and every leaf value
    - Restore original types via json.loads()
    - Verify PoP
    """

    def __init__(self, logger: Optional[AuditLogger] = None,
                 history_logger: Optional[CryptoHistoryLogger] = None):
        self.logger = logger or AuditLogger()
        self.history_logger = history_logger   # shared logger injected here

    def _hash_hex(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _hkp_key_bytes(self, hctx_hex: str, salt_hex: str) -> bytes:
        material = (hctx_hex + salt_hex).encode("utf-8")
        return hashlib.sha256(material).digest()

    def _to_canonical(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _dec_str(self, cipher: Fernet, s: str) -> str:
        return cipher.decrypt(s.encode("utf-8")).decode("utf-8")

    def _decrypt_structure(self, cipher: Fernet, obj: JsonObj) -> JsonObj:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for ek, v in obj.items():
                plain_key = self._dec_str(cipher, ek)
                out[plain_key] = self._decrypt_structure(cipher, v)
            return out

        if isinstance(obj, list):
            return [self._decrypt_structure(cipher, v) for v in obj]

        if isinstance(obj, str):
            try:
                val = self._dec_str(cipher, obj)
                # try to restore type
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return val
            except InvalidToken:
                return obj

        return obj

    def decrypt(self, packet: Dict[str, Any], *,
                hctx_hex: Optional[str] = None,
                salt_hex: Optional[str] = None) -> Dict[str, Any]:

        enc_map = packet.get("enc_map")
        pop = packet.get("pop")
        metadata = packet.get("metadata", {}) or {}

        role = metadata.get("role", "")
        epoch = metadata.get("epoch", "")
        kdf = metadata.get("kdf") or {}

        hctx_hex = hctx_hex or kdf.get("hctx")
        salt_hex = salt_hex or kdf.get("salt")

        if not (hctx_hex and salt_hex):
            raise ValueError("Missing HKP KDF material")

        fernet_key = base64.urlsafe_b64encode(self._hkp_key_bytes(hctx_hex, salt_hex))
        cipher = Fernet(fernet_key)

        plaintext_obj = self._decrypt_structure(cipher, enc_map)
        if not isinstance(plaintext_obj, dict):
            raise ValueError("Decryption produced non-dict root")

        canonical_plain = self._to_canonical(plaintext_obj)
        expected_pop = self._hash_hex(canonical_plain + role + epoch)
        if expected_pop != pop:
            self.logger.log("Decryptor", "PoPMismatch", {
                "expected": expected_pop, "got": pop, "role": role, "epoch": epoch
            })
            raise ValueError("PoP verification failed")
        
        self.history_logger.log_pair(packet["enc_map"], plaintext_obj)

        self.logger.log("Decryptor", "Decrypt", {
            "role": role, "epoch": epoch, "ok": True,
            "field_count": len(plaintext_obj)
        })
        return plaintext_obj


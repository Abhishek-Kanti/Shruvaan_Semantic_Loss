# praeceptor.py
import json
import math
import random
import os
from collections import deque
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Import existing project probes
from mimicus import run_mimicus
from probator import run_probator

# optional loggers
try:
    from audit_logger import AuditLogger
    from crypto_history_logger import CryptoHistoryLogger
except Exception:
    AuditLogger = None
    CryptoHistoryLogger = None

# ------------- Config / Globals -------------
PARAMS = ['obfuscation_depth', 'noise_scale', 'mask_rate', 'unicode_rate', 'length_jitter']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loss weights
BETA1, BETA2, BETA3 = 0.4, 0.5, 0.1
LAMBDA_DPO = 0.2
CHECKPOINT_PATH = "praeceptor_checkpoint.pt"

# --------- kernels ----------
class RBFKernel:
    def __init__(self, gamma=1.0): self.gamma = float(gamma)
    def __call__(self, x, y): return torch.exp(-self.gamma * ((x - y).pow(2).sum(dim=-1)))

class PolynomialKernel:
    def __init__(self, degree=2): self.degree = degree
    def __call__(self, x, y): return (1.0 + (x * y).sum(dim=-1)).pow(self.degree)

class MahalanobisKernel:
    def __init__(self): self.cov_inv = None
    def set_covariance(self, mat):
        eps = 1e-3
        try: self.cov_inv = torch.inverse(mat + eps * torch.eye(mat.size(0), device=mat.device))
        except Exception:
            diag = (mat.diag().abs() + eps).clone()
            self.cov_inv = torch.diag(1.0 / diag)
    def __call__(self, x, y):
        d = x - y
        if self.cov_inv is None: return torch.exp(-(d.pow(2).sum(dim=-1)))
        vi = torch.matmul(d, self.cov_inv); q = (vi * d).sum(dim=-1)
        return torch.exp(-q)

class SpectralKernel:
    def __init__(self, scale=1.0): self.scale = scale
    def __call__(self, x, y): return torch.cos(self.scale * (x - y)).mean(dim=-1)

# ------------ Controllers / Encoders -----------
class StateEncoder(nn.Module):
    def __init__(self, input_dim=18, hidden=128, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, emb_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class ActionEncoder(nn.Module):
    def __init__(self, input_dim=len(PARAMS), emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim), nn.ReLU())
    def forward(self, a): return self.net(a)

class HierarchicalPraeceptorController(nn.Module):
    def __init__(self, state_dim=18, emb_dim=128):
        super().__init__()
        self.enc = StateEncoder(input_dim=state_dim, emb_dim=emb_dim)
        self.local = nn.Linear(emb_dim, 64)
        self.global_ = nn.Linear(emb_dim, 64)
        self.mean = nn.Linear(64, len(PARAMS))
        self.logstd = nn.Linear(64, len(PARAMS))
        self.kweights_logits = nn.Parameter(torch.zeros(2))
    def forward(self, state):
        f = self.enc(state); l = self.local(f); g = self.global_(f)
        kw = torch.softmax(self.kweights_logits, dim=0); c = kw[0] * l + kw[1] * g
        mean = torch.tanh(self.mean(c)) * 0.25; logstd = self.logstd(c)
        logstd = torch.clamp(logstd, -6.0, 2.0); mean = torch.clamp(mean, -10.0, 10.0)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=2.0, neginf=-6.0)
        return mean, logstd

class KernelMixer(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__(); self.device = device
        self.term_logits = nn.Parameter(torch.zeros(6, device=device))
        self.rbf, self.poly, self.mahal, self.spec = RBFKernel(), PolynomialKernel(), MahalanobisKernel(), SpectralKernel()
    def get_weights(self): return torch.softmax(self.term_logits, dim=0)
    def entropy_regularizer(self):
        w = torch.softmax(self.term_logits, dim=0); return - (w * torch.log(w + 1e-12)).sum()
    def set_mahal_cov(self, cov): self.mahal.set_covariance(cov)
    def compute_hmk(self, ex, ey_pos, ey_neg):
        r_a, r_b = self.rbf(ex, ey_pos), self.rbf(ex, ey_neg)
        p_a, p_b = self.poly(ex, ey_pos), self.poly(ex, ey_neg)
        m_a, m_b = self.mahal(ex, ey_pos), self.mahal(ex, ey_neg)
        s_a, s_b = self.spec(ex, ey_pos), self.spec(ex, ey_neg)
        w = self.get_weights()
        local = w[2]*(r_a-r_b)+w[3]*(p_a-p_b); global_sim = w[4]*(s_a-s_b)+w[5]*(m_a-m_b)
        out = w[0]*local + w[1]*global_sim
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

# ---------- Experience / buffer ----------
class DPOExperience:
    def __init__(self, state, preferred_action, rejected_action, reward_diff):
        self.state, self.preferred_action, self.rejected_action, self.reward_diff = state.detach(), preferred_action.detach(), rejected_action.detach(), reward_diff
class ExperienceBuffer:
    def __init__(self, capacity=2000): self.buffer = deque(maxlen=capacity)
    def add(self, experience): self.buffer.append(experience)
    def sample(self, batch_size): return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self): return len(self.buffer)

# ---------- Cryptor cascade ----------
class CryptorCascade(nn.Module):
    def __init__(self, dim=len(PARAMS)):
        super().__init__(); self.theta = nn.Parameter(torch.zeros(dim, device=device))
    def forward(self): return torch.sigmoid(self.theta)

# ------------- Utilities: feature extraction -------------
def _rough_entropy(text: str) -> float:
    if not text: return 0.0
    counts = {}; [counts.setdefault(ch,0) or counts.update({ch:counts[ch]+1}) for ch in text]
    N = len(text); return float(-sum((v/N)*math.log((v/N)+1e-12) for v in counts.values()))

def _frac_digits(s: str) -> float: return (sum(1 for c in s if c.isdigit())) / max(1,len(s))
def _count_entities(decrypted: Dict[str, Any]) -> int: return len(decrypted.get("entities") or {})

# ---------------- Part 2: Praeceptor class + training loop + checkpoints ----------------

class Praeceptor:
    def __init__(self,
                 logger: Optional[Any] = None,
                 history_logger: Optional[Any] = None,
                 state_dim: int = 18,
                 lr: float = 1e-3,
                 device: str = device,
                 checkpoint_path: str = CHECKPOINT_PATH,
                 cryptor: Optional[Any] = None,
                 decryptor: Optional[Any] = None):
        self.logger = logger or (AuditLogger() if AuditLogger else None)
        self.history_logger = history_logger or (CryptoHistoryLogger() if CryptoHistoryLogger else None)
        self.device = device
        self.checkpoint_path = checkpoint_path

        # inject cryptor/decryptor (for training loops)
        self.cryptor_instance = cryptor
        self.decryptor_instance = decryptor

        # model pieces
        self.controller = HierarchicalPraeceptorController(state_dim=state_dim).to(self.device)
        self.action_encoder = ActionEncoder(input_dim=len(PARAMS)).to(self.device)
        self.km = KernelMixer(device=self.device).to(self.device)
        self.cryptor = CryptorCascade().to(self.device)

        # training buffers & optimizer
        self.exp_buffer = ExperienceBuffer(capacity=2000)
        params = (list(self.controller.parameters()) +
                  list(self.action_encoder.parameters()) +
                  list(self.km.parameters()) +
                  list(self.cryptor.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.batch_size = 32

        # running history
        self.recent_mimic = deque(maxlen=50)
        self.recent_prob = deque(maxlen=50)

        # load checkpoint if present
        self.load_checkpoint()

    # ------------------ internal helpers ------------------
    def _build_state(self, decrypted: Dict[str, Any], encrypted: Dict[str, Any], mimic_score: float, prob_score: float) -> torch.Tensor:
        enc_text = json.dumps(encrypted)
        plain_text = json.dumps(decrypted)

        len_enc = len(enc_text)
        num_enc_tokens = len(set(enc_text.split()))
        enc_digit_frac = _frac_digits(enc_text)
        enc_entropy = _rough_entropy(enc_text)

        len_plain = len(plain_text)
        num_entities = _count_entities(decrypted)
        has_amount = 1.0 if (decrypted.get("entities") or {}).get("amount") else 0.0
        has_account = 1.0 if any(k for k in (decrypted.get("entities") or {}) if "account" in k.lower()) else 0.0
        digits_plain_frac = _frac_digits(plain_text)
        plain_entropy = _rough_entropy(plain_text)

        hist_mimic_mean = float((sum(self.recent_mimic) / len(self.recent_mimic)) if self.recent_mimic else 0.0)
        hist_prob_mean = float((sum(self.recent_prob) / len(self.recent_prob)) if self.recent_prob else 0.0)

        features = [
            float(len_enc) / 1000.0,
            float(num_enc_tokens) / 100.0,
            enc_digit_frac,
            enc_entropy / 10.0,
            float(len_plain) / 1000.0,
            float(num_entities) / 10.0,
            has_amount,
            has_account,
            digits_plain_frac,
            plain_entropy / 10.0,
            float(mimic_score),
            float(prob_score),
            hist_mimic_mean,
            hist_prob_mean,
            1.0 if decrypted.get("intent") else 0.0,
            1.0 if decrypted.get("role") else 0.0,
            0.0,
            0.0,
        ]
        st = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        return st

    def _train_step_dpo(self, train_batch: List[DPOExperience], alpha: float = 0.1, lambda_js: float = 0.01, ent_coeff: float = 1e-3) -> torch.Tensor:
        if not train_batch:
            return torch.tensor(0.0, device=self.device)
        states = torch.cat([exp.state for exp in train_batch], dim=0)
        preferred_actions = torch.cat([exp.preferred_action for exp in train_batch], dim=0)
        rejected_actions = torch.cat([exp.rejected_action for exp in train_batch], dim=0)

        mean, logstd = self.controller(states)
        std = torch.exp(logstd).clamp(min=1e-6, max=1e2)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e2, neginf=-1e2)
        std = torch.nan_to_num(std, nan=1e-3, posinf=1e2, neginf=1e-3)

        dist = Normal(mean, std)
        try:
            log_prob_pref = dist.log_prob(preferred_actions).sum(dim=-1)
            log_prob_rej = dist.log_prob(rejected_actions).sum(dim=-1)
        except Exception:
            return torch.tensor(0.0, device=self.device)

        delta = log_prob_pref - log_prob_rej

        ex = self.controller.enc(states)
        ey_pref = self.action_encoder(preferred_actions)
        ey_rej = self.action_encoder(rejected_actions)

        cat = torch.cat([ey_pref, ey_rej], dim=0)
        if cat.size(0) > 1:
            mu = cat.mean(dim=0, keepdim=True)
            Xc = cat - mu
            cov = (Xc.t() @ Xc) / (cat.size(0) - 1)
            try:
                self.km.set_mahal_cov(cov.detach())
            except Exception:
                pass

        kernel_sim = self.km.compute_hmk(ex, ey_pref, ey_rej)
        kernel_sim = torch.nan_to_num(kernel_sim, nan=0.0, posinf=1.0, neginf=-1.0)
        delta = delta + alpha * kernel_sim

        dpo_loss = -F.logsigmoid(delta).mean()
        js_reg = torch.tensor(0.0, device=self.device)
        ent_reg = self.km.entropy_regularizer()
        total_dpo = dpo_loss + lambda_js * js_reg - ent_coeff * ent_reg
        if torch.isnan(total_dpo) or torch.isinf(total_dpo):
            return torch.tensor(0.0, device=self.device)
        return total_dpo

    # ------------------ core API: single-step policy update ------------------
    def step_update(self,
                    decrypted: Dict[str, Any],
                    encrypted: Dict[str, Any],
                    collect_preference: bool = True,
                    dpo_collect_prob: float = 0.1) -> Dict[str, Any]:
        """
        Single-step pipeline: probes -> state -> sample theta -> compute leakage loss -> update weights
        Returns: debug dict with theta, leakage numbers, diagnostics.
        """
        # 1. probes
        mimic_result = run_mimicus(decrypted, encrypted, logger=self.logger, history_logger=self.history_logger)
        prob_result = run_probator(decrypted, encrypted, logger=self.logger)

        mimic_score = float(mimic_result.get("leakage_score", mimic_result.get("leakage", 0.0)))
        prob_score = float(prob_result.get("Rprob", prob_result.get("rprob", 0.0)))

        # store history
        self.recent_mimic.append(mimic_score)
        self.recent_prob.append(prob_score)

        # 2. state
        state = self._build_state(decrypted, encrypted, mimic_score, prob_score)

        # 3. sample action (policy)
        mean, logstd = self.controller(state)
        std = torch.exp(logstd).clamp(min=1e-6, max=1e2)
        dist = Normal(mean, std)
        action = dist.rsample()
        # proposed theta values (0..1)
        theta_proposed = torch.sigmoid(action).squeeze(0).detach().cpu().numpy().tolist()

        # 4. leakage loss (paper formula)
        mimic_loss = torch.tensor(float(mimic_score), device=self.device)
        risk_loss = torch.tensor(float(prob_score), device=self.device)
        entropy_term = -dist.entropy().mean()
        leakage_loss_val = BETA1 * mimic_loss + BETA2 * risk_loss + BETA3 * entropy_term

        # 5. gather DPO preference example (approximate)
        dpo_loss_val = torch.tensor(0.0, device=self.device)
        if collect_preference and torch.rand(1).item() < dpo_collect_prob:
            alt_action = dist.rsample()
            # heuristics for preference: prefer smaller action magnitude (regularizer) while same leakage
            reward_A = float((mimic_score + prob_score) - 0.01 * action.abs().sum().item())
            reward_B = float((mimic_score + prob_score) - 0.01 * alt_action.abs().sum().item())
            if reward_A >= reward_B:
                exp = DPOExperience(state, action, alt_action, reward_A - reward_B)
            else:
                exp = DPOExperience(state, alt_action, action, reward_B - reward_A)
            self.exp_buffer.add(exp)

        # 6. DPO batch update (if available)
        if len(self.exp_buffer) > self.batch_size:
            batch = self.exp_buffer.sample(self.batch_size)
            dpo_loss_val = self._train_step_dpo(batch)

        # 7. combine and update
        total_loss = leakage_loss_val + LAMBDA_DPO * dpo_loss_val
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if self.logger:
                try: self.logger.log("Praeceptor", "SkipUpdate", {"reason": "NaN/Inf total_loss"})
                except Exception: pass
            # return current policy's sampled theta for instrumentation
            return {"theta": theta_proposed, "leakage": {"mimic": mimic_score, "prob": prob_score}}

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.controller.parameters()) + list(self.action_encoder.parameters()) + list(self.km.parameters()) + list(self.cryptor.parameters()), max_norm=1.0)
        self.optimizer.step()

        # return current cryptor theta for external use (cryptor.theta is trainable param)
        theta_current = torch.sigmoid(self.cryptor.theta).detach().cpu().numpy().tolist()
        out = {
            "theta": theta_current,
            "leakage": {"mimic": mimic_score, "prob": prob_score},
            "nli_metrics": mimic_result.get("components", {}),
            "prob_metrics": prob_result,
            "state": state.cpu().numpy().tolist(),
            "total_loss": float(total_loss.detach().cpu().numpy()),
            "dpo_loss": float(dpo_loss_val.detach().cpu().numpy()) if isinstance(dpo_loss_val, torch.Tensor) else float(dpo_loss_val),
        }
        if self.logger:
            try: self.logger.log("Praeceptor", "StepUpdate", out)
            except Exception: pass
        return out

    # ------------------ inner-loop cipher train that runs until 'safe' or budget exhausted ------------------
    def train_until_safe(self,
                        normalized_instruction: Dict[str, Any],
                        cryptor: Optional[Any] = None,
                        decryptor: Optional[Any] = None,
                        safe_threshold: float = 0.25,
                        max_steps: int = 200,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Inner-loop optimization: repeatedly re-encrypts with current theta, re-decrypts,
        probes leakage, and updates weights until leakage <= safe_threshold or max_steps reached.

        Args:
            normalized_instruction: the plaintext dict (normalized)
            cryptor: external Cryptor (optional, else uses self.cryptor_instance)
            decryptor: external Decryptor (optional, else uses self.decryptor_instance)
            safe_threshold: target leakage score
            max_steps: max optimization iterations
            verbose: print progress if True

        Returns:
            dict with {"success", "final_theta", "history"}.
        """
        # use internal if external not supplied
        cryptor = cryptor or self.cryptor_instance
        decryptor = decryptor or self.decryptor_instance

        if cryptor is None or not hasattr(cryptor, "encrypt"):
            raise RuntimeError("train_until_safe requires a Cryptor with encrypt()")

        if decryptor is None or not hasattr(decryptor, "decrypt"):
            raise RuntimeError("train_until_safe requires a Decryptor with decrypt()")

        hist = []
        external_set_theta = hasattr(cryptor, "set_theta")

        for step in range(max_steps):
            # 1. sync external cryptor theta with our learned param
            current_theta = torch.sigmoid(self.cryptor.theta).detach().cpu().numpy().tolist()
            if external_set_theta:
                try:
                    cryptor.set_theta(current_theta)
                except Exception:
                    pass

            # 2. encrypt + decrypt
            try:
                packet = cryptor.encrypt(normalized_instruction)
            except Exception as e:
                return {"success": False, "reason": "encrypt_failed", "error": str(e)}

            try:
                recovered = decryptor.decrypt(packet)
            except Exception as e:
                if self.logger:
                    try: self.logger.log("Praeceptor", "DecryptFail", {"step": step, "error": str(e)})
                    except Exception: pass
                return {
                    "success": False, 
                    "reason": "decrypt_failed", 
                    "final_theta": torch.sigmoid(self.cryptor.theta).detach().cpu().numpy().tolist(),
                    "error": str(e)
                }

            # 3. probes
            mimic_res = run_mimicus(recovered, packet, logger=self.logger, history_logger=self.history_logger)
            prob_res = run_probator(recovered, packet, logger=self.logger)
            mimic_score = float(mimic_res.get("leakage_score", 0.0))
            prob_score = float(prob_res.get("Rprob", 0.0))
            combined_leak = BETA1 * mimic_score + BETA2 * prob_score

            hist.append({
                "step": step,
                "mimic": mimic_score,
                "prob": prob_score,
                "combined": combined_leak,
                "theta": current_theta
            })

            if verbose:
                print(f"[Inner {step}] mimic={mimic_score:.4f}, prob={prob_score:.4f}, combined={combined_leak:.4f}")

            if combined_leak <= safe_threshold:
                if self.logger:
                    try: self.logger.log("Praeceptor", "InnerSafe",
                                        {"step": step, "combined": combined_leak, "theta": current_theta})
                    except Exception: pass
                return {
                    "success": True, 
                    "final_theta": current_theta, 
                    "history": hist
                }

            # 4. build state + compute leakage loss
            state = self._build_state(recovered, packet, mimic_score, prob_score)
            mean, logstd = self.controller(state)
            std = torch.exp(logstd).clamp(min=1e-6, max=1e2)
            dist = Normal(mean, std)
            action = dist.rsample()

            mimic_loss = torch.tensor(mimic_score, device=self.device)
            risk_loss = torch.tensor(prob_score, device=self.device)
            entropy_term = -dist.entropy().mean()
            leakage_loss_val = BETA1 * mimic_loss + BETA2 * risk_loss + BETA3 * entropy_term

            dpo_loss_val = torch.tensor(0.0, device=self.device)
            if len(self.exp_buffer) > self.batch_size:
                batch = self.exp_buffer.sample(self.batch_size)
                dpo_loss_val = self._train_step_dpo(batch)

            total_loss = leakage_loss_val + LAMBDA_DPO * dpo_loss_val

            # 5. gradient step
            self.optimizer.zero_grad()
            try:
                total_loss.backward()
            except Exception as e:
                if self.logger:
                    try: self.logger.log("Praeceptor", "BackwardFail", {"step": step, "err": str(e)})
                    except Exception: pass
                return {
                        "success": False, 
                        "reason": "backward_fail", 
                        "final_theta": torch.sigmoid(self.cryptor.theta).detach().cpu().numpy().tolist(),
                        "error": str(e)
                    }

            torch.nn.utils.clip_grad_norm_(
                list(self.controller.parameters()) +
                list(self.action_encoder.parameters()) +
                list(self.km.parameters()) +
                list(self.cryptor.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

        # exhausted loop
        if self.logger:
            try: self.logger.log("Praeceptor", "InnerExhaust",
                                {"max_steps": max_steps, "last": hist[-1] if hist else None})
            except Exception: pass
        return {
            "success": False,
            "final_theta": torch.sigmoid(self.cryptor.theta).detach().cpu().numpy().tolist(),
            "history": hist
        }


    # ------------------ checkpointing ------------------
    def save_checkpoint(self, path: Optional[str] = None):
        path = path or self.checkpoint_path
        try:
            state = {
                "controller": self.controller.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "km": self.km.state_dict(),
                "cryptor": self.cryptor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, path)
            if self.logger:
                try: self.logger.log("Praeceptor", "SaveCheckpoint", {"path": path})
                except Exception: pass
            return True
        except Exception as e:
            if self.logger:
                try: self.logger.log("Praeceptor", "SaveCheckpointFail", {"error": str(e)})
                except Exception: pass
            return False

    def load_checkpoint(self, path: Optional[str] = None):
        path = path or self.checkpoint_path
        if not os.path.exists(path):
            return False
        try:
            state = torch.load(path, map_location=self.device)
            self.controller.load_state_dict(state.get("controller", {}))
            self.action_encoder.load_state_dict(state.get("action_encoder", {}))
            self.km.load_state_dict(state.get("km", {}))
            self.cryptor.load_state_dict(state.get("cryptor", {}))
            if "optimizer" in state:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                except Exception:
                    # optimizer states can be device-specific; ignore if fails
                    pass
            if self.logger:
                try: self.logger.log("Praeceptor", "LoadCheckpoint", {"path": path})
                except Exception: pass
            return True
        except Exception as e:
            if self.logger:
                try: self.logger.log("Praeceptor", "LoadCheckpointFail", {"error": str(e)})
                except Exception: pass
            return False

# ------------------ convenience wrapper ------------------
def run_praeceptor_step(decrypted, encrypted, logger=None, history_logger=None):
    pre = Praeceptor(logger=logger, history_logger=history_logger)
    return pre.step_update(decrypted, encrypted)

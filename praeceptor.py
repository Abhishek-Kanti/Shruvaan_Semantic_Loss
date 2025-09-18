# praeceptor.py
import json
import math
import random
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

# Import existing project probes
from mimicus import run_mimicus
from probator import run_probator

# optional logger types in your repo
try:
    from audit_logger import AuditLogger
    from crypto_history_logger import CryptoHistoryLogger
except Exception:
    AuditLogger = None
    CryptoHistoryLogger = None

# ------------- Config / Globals -------------
PARAMS = ['obfuscation_depth', 'noise_scale', 'mask_rate', 'unicode_rate', 'length_jitter']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loss weights (paper suggestion)
BETA1, BETA2, BETA3 = 0.4, 0.5, 0.1
LAMBDA_DPO = 0.2

# --------- small kernel implementations (local/global mixture used in DPO) ----------
class RBFKernel:
    def __init__(self, gamma=1.0): self.gamma = float(gamma)
    def __call__(self, x, y):
        return torch.exp(-self.gamma * ((x - y).pow(2).sum(dim=-1)))

class PolynomialKernel:
    def __init__(self, degree=2): self.degree = degree
    def __call__(self, x, y):
        return (1.0 + (x * y).sum(dim=-1)).pow(self.degree)

class MahalanobisKernel:
    def __init__(self):
        self.cov_inv = None
    def set_covariance(self, mat):
        eps = 1e-3
        try:
            self.cov_inv = torch.inverse(mat + eps * torch.eye(mat.size(0), device=mat.device))
        except Exception:
            diag = (mat.diag().abs() + eps).clone()
            self.cov_inv = torch.diag(1.0 / diag)
    def __call__(self, x, y):
        d = x - y
        if self.cov_inv is None:
            return torch.exp(-(d.pow(2).sum(dim=-1)))
        vi = torch.matmul(d, self.cov_inv)
        q = (vi * d).sum(dim=-1)
        return torch.exp(-q)

class SpectralKernel:
    def __init__(self, scale=1.0): self.scale = scale
    def __call__(self, x, y):
        return torch.cos(self.scale * (x - y)).mean(dim=-1)

# ------------ Controllers / Encoders -----------
class StateEncoder(nn.Module):
    def __init__(self, input_dim=18, hidden=128, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
            nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class ActionEncoder(nn.Module):
    def __init__(self, input_dim=len(PARAMS), emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
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
        f = self.enc(state)
        l = self.local(f)
        g = self.global_(f)
        kw = torch.softmax(self.kweights_logits, dim=0)
        c = kw[0] * l + kw[1] * g
        mean = torch.tanh(self.mean(c)) * 0.25
        logstd = self.logstd(c)
        logstd = torch.clamp(logstd, min=-6.0, max=2.0)
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=2.0, neginf=-6.0)
        return mean, logstd

class KernelMixer(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.term_logits = nn.Parameter(torch.zeros(6, device=device))
        self.rbf = RBFKernel()
        self.poly = PolynomialKernel()
        self.mahal = MahalanobisKernel()
        self.spec = SpectralKernel()
    def get_weights(self):
        return torch.softmax(self.term_logits, dim=0)
    def entropy_regularizer(self):
        w = torch.softmax(self.term_logits, dim=0)
        return - (w * torch.log(w + 1e-12)).sum()
    def set_mahal_cov(self, cov):
        self.mahal.set_covariance(cov)
    def compute_hmk(self, ex, ey_pos, ey_neg):
        r_a = self.rbf(ex, ey_pos); r_b = self.rbf(ex, ey_neg)
        p_a = self.poly(ex, ey_pos); p_b = self.poly(ex, ey_neg)
        m_a = self.mahal(ex, ey_pos); m_b = self.mahal(ex, ey_neg)
        s_a = self.spec(ex, ey_pos); s_b = self.spec(ex, ey_neg)
        w = self.get_weights()
        local = w[2] * (r_a - r_b) + w[3] * (p_a - p_b)
        global_sim = w[4] * (s_a - s_b) + w[5] * (m_a - m_b)
        out = w[0] * local + w[1] * global_sim
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out

# ---------- Experience / buffer ----------
class DPOExperience:
    def __init__(self, state, preferred_action, rejected_action, reward_diff):
        self.state = state.detach()
        self.preferred_action = preferred_action.detach()
        self.rejected_action = rejected_action.detach()
        self.reward_diff = reward_diff

class ExperienceBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)
    def add(self, experience): self.buffer.append(experience)
    def sample(self, batch_size): 
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self): return len(self.buffer)

# ---------- Cryptor cascade object (holds theta param) ----------
class CryptorCascade(nn.Module):
    def __init__(self, dim=len(PARAMS)):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dim, device=device))
    def forward(self):
        return torch.sigmoid(self.theta)   # values in (0,1)

# ------------- Utilities: feature extraction -------------
def _rough_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    N = len(text)
    ent = 0.0
    for v in counts.values():
        p = v / N
        ent -= p * math.log(p + 1e-12)
    return float(ent)

def _frac_digits(s: str) -> float:
    if not s:
        return 0.0
    digits = sum(1 for c in s if c.isdigit())
    return digits / max(1, len(s))

def _count_entities(decrypted: Dict[str, Any]) -> int:
    ents = decrypted.get("entities") or {}
    return len(ents)

# ---------- Preceptor main class ----------
class Praeceptor:
    def __init__(self,
                 logger: Optional[Any] = None,
                 history_logger: Optional[Any] = None,
                 state_dim: int = 18,
                 lr: float = 1e-3,
                 device: str = device):
        self.logger = logger or (AuditLogger() if AuditLogger else None)
        self.history_logger = history_logger or (CryptoHistoryLogger() if CryptoHistoryLogger else None)
        self.device = device

        # model pieces
        self.controller = HierarchicalPraeceptorController(state_dim=state_dim).to(self.device)
        self.action_encoder = ActionEncoder(input_dim=len(PARAMS)).to(self.device)
        self.km = KernelMixer(device=self.device).to(self.device)
        self.cryptor = CryptorCascade().to(self.device)

        # training buffers & optimizer
        self.exp_buffer = ExperienceBuffer(capacity=2000)
        self.optimizer = torch.optim.Adam(list(self.controller.parameters()) +
                                          list(self.action_encoder.parameters()) +
                                          list(self.km.parameters()) +
                                          list(self.cryptor.parameters()), lr=lr)
        self.batch_size = 32

        # simple running history aggregates
        self.recent_mimic = deque(maxlen=50)
        self.recent_prob = deque(maxlen=50)

    # ---- build numeric state from decrypted + encrypted + signals ----
    def _build_state(self,
                     decrypted: Dict[str, Any],
                     encrypted: Dict[str, Any],
                     mimic_score: float,
                     prob_score: float) -> torch.Tensor:
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
            float(len_enc) / 1000.0,      # scale down
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
            0.0,  # reserved
            0.0,  # reserved
        ]
        st = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,18]
        return st

    # -------- DPO train_step (adapted & robust) --------
    def _train_step_dpo(self, train_batch: List[DPOExperience]) -> torch.Tensor:
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

        delta = log_prob_pref - log_prob_rej  # [B]

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
        delta = delta + 0.1 * kernel_sim  # small kernel term

        # dpo loss
        dpo_loss = -F.logsigmoid(delta).mean()
        # JS reg with a copy reference (optional; here we omit ref controller for simplicity)
        js_reg = torch.tensor(0.0, device=self.device)
        ent_reg = self.km.entropy_regularizer()
        total_dpo = dpo_loss + 0.01 * js_reg - 1e-3 * ent_reg
        if torch.isnan(total_dpo) or torch.isinf(total_dpo):
            return torch.tensor(0.0, device=self.device)
        return total_dpo

    # -------- single step update (primary API) -----------
    def step_update(self,
                    decrypted: Dict[str, Any],
                    encrypted: Dict[str, Any],
                    collect_preference: bool = True) -> Dict[str, Any]:
        """
        Run one step: compute leakage via Mimicus & Probator, build state,
        sample action, compute loss and update controller+cryptor.

        Returns dict: {"theta": list, "leakage": {...}, "state": tensor}
        """
        # 1. call probes
        print("\n=== Mimicus Probe ===")
        mimic_result = run_mimicus(decrypted, encrypted, logger=self.logger, history_logger=self.history_logger)
        print(json.dumps(mimic_result, indent=2))
        
        print("\n=== Probator Probe ===")
        prob_result = run_probator(decrypted, encrypted, logger=self.logger)
        print(json.dumps(prob_result, indent=2))

        mimic_score = float(mimic_result.get("leakage_score", mimic_result.get("leakage", 0.0)))
        prob_score = float(prob_result.get("Rprob", prob_result.get("rprob", 0.0)))

        # push to running history
        self.recent_mimic.append(mimic_score)
        self.recent_prob.append(prob_score)

        # 2. build state
        state = self._build_state(decrypted, encrypted, mimic_score, prob_score)

        # 3. controller forward -> distribution over actions
        mean, logstd = self.controller(state)
        std = torch.exp(logstd).clamp(min=1e-6, max=1e2)
        dist = Normal(mean, std)
        action = dist.rsample()  # sample (reparameterized)
        # action is an unconstrained vector; we'll squash to [0,1] via sigmoid for theta
        theta = torch.sigmoid(action).squeeze(0).detach().cpu().numpy().tolist()

        # 4. compute leakage loss components (paper formula)
        # mimic_loss use Mimicus leakage; risk_loss use Probator Rprob;
        mimic_loss = torch.tensor(float(mimic_score), device=self.device)
        risk_loss = torch.tensor(float(prob_score), device=self.device)
        # entropy term: negative expected entropy (paper used -E[H])
        entropy_term = -dist.entropy().mean()  # if entropy small -> negative small? we penalize low entropy
        # total leakage-aware loss (to minimize)
        leakage_loss_val = BETA1 * mimic_loss + BETA2 * risk_loss + BETA3 * entropy_term

        # 5. possibly collect DPO preference (sample alternative action)
        dpo_loss_val = torch.tensor(0.0, device=self.device)
        if collect_preference and random.random() < 0.1:
            alt_action = dist.rsample()
            # evaluate alt via mocked / same leakage proxies: call probes for alt? expensive.
            # Instead approximate: simulate alt by small perturbation -> but easier is to reuse current
            # (in real system you would re-run decrypt/encrypt with alt theta and compute mimic/prob)
            # We'll approximate by comparing loss under small noise versions:
            # reward A (current) and B (alt)
            alt_theta = torch.sigmoid(alt_action)
            # For demonstration: compute reward proxies using action norms (heuristic)
            reward_A = float((mimic_score * 1.0) + (prob_score * 1.0) - 0.01 * action.abs().sum().item())
            reward_B = float((mimic_score * 1.0) + (prob_score * 1.0) - 0.01 * alt_action.abs().sum().item())
            if reward_A >= reward_B:
                exp = DPOExperience(state, action, alt_action, reward_A - reward_B)
            else:
                exp = DPOExperience(state, alt_action, action, reward_B - reward_A)
            self.exp_buffer.add(exp)

        # 6. DPO batch update
        if len(self.exp_buffer) > self.batch_size:
            batch = self.exp_buffer.sample(self.batch_size)
            dpo_loss_val = self._train_step_dpo(batch)

        # 7. combine losses and step optimizer
        total_loss = leakage_loss_val + LAMBDA_DPO * dpo_loss_val
        # defensive guards
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if self.logger:
                try:
                    self.logger.log("Preceptor", "SkipUpdate", {"reason": "NaN/Inf total_loss"})
                except Exception:
                    pass
            return {"theta": theta, "leakage": {"mimic": mimic_score, "prob": prob_score}, "state": state.cpu().numpy()}

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.controller.parameters()) +
                                       list(self.action_encoder.parameters()) +
                                       list(self.km.parameters()) +
                                       list(self.cryptor.parameters()), max_norm=1.0)
        self.optimizer.step()

        # 8. return theta and useful debug info
        theta_out = torch.sigmoid(self.cryptor.theta).detach().cpu().numpy().tolist()
        out = {
            "theta": theta_out,
            "leakage": {"mimic": mimic_score, "prob": prob_score},
            "nli_metrics": mimic_result.get("components", {}),
            "prob_metrics": prob_result,
            "state": state.cpu().numpy().tolist(),
            "total_loss": float(total_loss.detach().cpu().numpy()),
            "dpo_loss": float(dpo_loss_val.detach().cpu().numpy()) if isinstance(dpo_loss_val, torch.Tensor) else float(dpo_loss_val),
        }
        # log
        if self.logger:
            try:
                self.logger.log("Preceptor", "StepUpdate", out)
            except Exception:
                pass

        return out

# convenience function for outside usage
def run_praeceptor_step(decrypted, encrypted, logger=None, history_logger=None):
    pre = Praeceptor(logger=logger, history_logger=history_logger)
    return pre.step_update(decrypted, encrypted)

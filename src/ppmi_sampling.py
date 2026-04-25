"""
Flexible LLaDA reverse-diffusion sampling with configurable unmasking-order scoring.

Two scoring modes:
  "confidence"  — max softmax probability (identical to Algorithm 5 baseline)
  "ppmi"        — PPMI-inspired position score:
                      ppmi_score(i) = Σ_{j≠i, j masked} A[i,j] · confidence(j)
                  A is the mean attention weight matrix over layers and heads.

Attention weights are captured via SDPACapture — a wrapper around
F.scaled_dot_product_attention that must be installed BEFORE any
TALMASHookManager is created (so TALMAS's internal original_sdpa reference
points through this wrapper, giving us the TALMAS-biased weights for free).
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F

from src.config import SamplingConfig


# ---------------------------------------------------------------------------
# Attention weight capture
# ---------------------------------------------------------------------------

class SDPACapture:
    """
    Wraps F.scaled_dot_product_attention to capture per-layer attention weights.

    Install BEFORE creating any TALMASHookManager when both PPMI and TALMAS
    are needed.  TALMAS captures F.sdpa as its original_sdpa at __init__ time;
    installing SDPACapture first makes TALMAS's biased SDPA call back through
    here, so captured weights reflect the TALMAS bias automatically.

    Usage:
        capture = SDPACapture().install()
        # ... create TALMASHookManager (if any) ...
        # Per forward pass:
        capture.clear()
        model(input_ids=...)
        layers = capture.get()   # list of (S, S) tensors, one per layer call
        # At end of evaluation:
        capture.uninstall()
    """

    def __init__(self):
        self._true_sdpa = None
        self._captured: list = []

    @property
    def installed(self) -> bool:
        return self._true_sdpa is not None

    def install(self) -> "SDPACapture":
        self._true_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._forward
        return self

    def uninstall(self) -> None:
        if self._true_sdpa is not None:
            F.scaled_dot_product_attention = self._true_sdpa
            self._true_sdpa = None

    def clear(self) -> None:
        self._captured.clear()

    def get(self) -> list:
        """Return list of (S, S) mean-over-heads attention tensors, one per layer."""
        return list(self._captured)

    def _forward(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, **kw):
        # Actual computation through the real (or TALMAS-biased) SDPA
        output = self._true_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw
        )

        # Compute attention weights separately for PPMI scoring only.
        # query/key shape: (B, H, S, d_k)
        with torch.no_grad():
            scale = kw.get("scale") or (query.shape[-1] ** -0.5)
            scores = (query.float() @ key.float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask.float()
            if is_causal:
                q_len, k_len = query.shape[-2], key.shape[-2]
                causal = torch.ones(q_len, k_len, dtype=torch.bool,
                                    device=query.device).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            # Average over heads immediately to keep memory low → (S, S)
            attn_w = torch.softmax(scores, dim=-1).mean(dim=1).squeeze(0)
            self._captured.append(attn_w.detach())

        return output


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.inference_mode()
def flexible_remasking_sample(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (1, prompt_len) — already on device
    cfg: SamplingConfig,
    device: torch.device,
    mask_token_id: int,
    eos_token_id: int,
    hook_manager=None,                  # TALMASHookManager | None
    scoring: Literal["confidence", "ppmi"] = "confidence",
    layer_agg: Literal["all", "last_half"] = "all",
    sdpa_capture: SDPACapture = None,   # required when scoring="ppmi"
) -> torch.Tensor:
    """
    Reverse-diffusion sampling with configurable unmasking-order scoring.

    When scoring="ppmi", sdpa_capture must be an installed SDPACapture instance.
    The capture is cleared before each forward pass and aggregated afterwards.

    Returns the generated token IDs as a 1-D tensor (response only).
    """
    L = cfg.generation_length
    N = cfg.steps
    use_ppmi = (scoring == "ppmi")

    response  = torch.full((1, L), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, response], dim=1)
    prompt_len = prompt_ids.shape[1]

    timesteps = torch.linspace(1.0, 1.0 / N, N, device=device)

    for t in timesteps:
        s     = max((t - 1.0 / N).item(), 0.0)
        t_val = t.item()

        if hook_manager is not None:
            mask_positions = (input_ids == mask_token_id)
            hook_manager.set_state(r_t=t_val, mask_positions=mask_positions)

        # Clear capture buffer before each forward pass
        if use_ppmi and sdpa_capture is not None:
            sdpa_capture.clear()

        outputs = model(input_ids=input_ids)

        response_logits = outputs.logits[0, prompt_len:, :]
        probs      = F.softmax(response_logits, dim=-1)
        pred_ids   = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        if cfg.zero_eos_confidence:
            eos_mask   = (pred_ids == eos_token_id)
            confidence = confidence.masked_fill(eos_mask, 0.0)

        # -----------------------------------------------------------------
        # Compute unmasking-order score
        # -----------------------------------------------------------------
        if use_ppmi and sdpa_capture is not None:
            captured = sdpa_capture.get()  # list of (S, S) per layer
            if captured:
                if layer_agg == "last_half":
                    captured = captured[len(captured) // 2:]
                # Mean over layers → (S, S), then slice response block → (L, L)
                A      = torch.stack(captured, dim=0).mean(dim=0)
                A_resp = A[prompt_len:, prompt_len:].clone()
                del A, captured

                # ppmi_score(i) = Σ_{j≠i, j masked} A_resp[i,j] · confidence[j]
                j_mask   = (input_ids[0, prompt_len:] == mask_token_id).float()
                off_diag = 1.0 - torch.eye(L, device=device)
                score = (
                    A_resp
                    * confidence.unsqueeze(0)   # broadcast: (1, L) → (L, L)
                    * j_mask.unsqueeze(0)        # zero out unmasked j
                    * off_diag                   # zero out diagonal
                ).sum(dim=-1)                    # (L,)

                if cfg.zero_eos_confidence:
                    score = score.masked_fill(eos_mask, 0.0)

                del A_resp
            else:
                score = confidence              # fallback if capture empty
        else:
            score = confidence

        # -----------------------------------------------------------------
        # Lock already-unmasked positions (always keep them selected)
        # -----------------------------------------------------------------
        current_response = input_ids[0, prompt_len:]
        already_unmasked = (current_response != mask_token_id)
        score    = score.masked_fill(already_unmasked, float("inf"))
        pred_ids = torch.where(already_unmasked, current_response, pred_ids)

        n_unmask = max(0, min(math.floor(L * (1.0 - s)), L))

        _, top_indices = torch.topk(score, k=n_unmask, largest=True)
        new_response = torch.full((L,), mask_token_id, dtype=torch.long, device=device)
        new_response[top_indices] = pred_ids[top_indices]

        input_ids[0, prompt_len:] = new_response

    # Drop everything after the first EOS token
    final = input_ids[0, prompt_len:]
    eos_positions = (final == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        final = final[: eos_positions[0]]

    return final

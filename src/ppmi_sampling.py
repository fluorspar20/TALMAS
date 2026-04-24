"""
Flexible LLaDA reverse-diffusion sampling with configurable unmasking-order scoring.

Two scoring modes are available:
  "confidence"  — max softmax probability (identical to Algorithm 5 baseline)
  "ppmi"        — PPMI-inspired position score using attention weights as a
                  coupling proxy between masked positions:

                      ppmi_score(i) = Σ_{j≠i, j masked} A[i,j] · confidence(j)

                  A[i,j] is the mean attention weight from position i to j,
                  averaged across all (or the top-half) layers and all heads.
                  A high score means position i strongly attends to confident
                  masked peers — it is "load-bearing" for resolving others.

Both modes work with or without a TALMAS hook_manager.  When TALMAS is active
and scoring="ppmi", the captured attention weights reflect the TALMAS-biased
distribution (bias is applied inside the model before the attention softmax).
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F

from src.config import SamplingConfig


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
) -> torch.Tensor:
    """
    Reverse-diffusion sampling with configurable unmasking-order scoring.

    When scoring="ppmi" the model is called with output_attentions=True and
    attention weights are aggregated across layers and heads to build the PPMI
    position score.  eager attn_implementation is required (enforced externally
    via load_model_and_tokenizer(eager_attn=True)).

    Returns the generated token IDs as a 1-D tensor (response portion only).
    """
    L = cfg.generation_length
    N = cfg.steps
    use_ppmi = (scoring == "ppmi")

    # Initialise: fully masked response appended to prompt
    response  = torch.full((1, L), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, response], dim=1)  # (1, prompt_len + L)
    prompt_len = prompt_ids.shape[1]

    # Uniform time-steps: t goes from 1 → 1/N
    timesteps = torch.linspace(1.0, 1.0 / N, N, device=device)

    for t in timesteps:
        s     = max((t - 1.0 / N).item(), 0.0)
        t_val = t.item()

        # TALMAS: update hook state before each forward pass
        if hook_manager is not None:
            mask_positions = (input_ids == mask_token_id)
            hook_manager.set_state(r_t=t_val, mask_positions=mask_positions)

        # Forward pass — request attentions only when needed for PPMI
        outputs = model(input_ids=input_ids, output_attentions=use_ppmi)

        response_logits = outputs.logits[0, prompt_len:, :]  # (L, vocab)
        probs      = F.softmax(response_logits, dim=-1)
        pred_ids   = probs.argmax(dim=-1)       # (L,)
        confidence = probs.max(dim=-1).values   # (L,)

        # Optional: suppress EOS positions so they are not unmasked too early
        if cfg.zero_eos_confidence:
            eos_mask   = (pred_ids == eos_token_id)
            confidence = confidence.masked_fill(eos_mask, 0.0)

        # -----------------------------------------------------------------
        # Compute the unmasking-order score
        # -----------------------------------------------------------------
        if use_ppmi:
            attn_layers = outputs.attentions  # tuple of (1, H, S, S) per layer
            if layer_agg == "last_half":
                attn_layers = attn_layers[len(attn_layers) // 2:]

            # Mean over selected layers, then mean over heads → (S, S)
            A      = torch.stack(list(attn_layers), dim=0).mean(0).mean(1).squeeze(0)
            A_resp = A[prompt_len:, prompt_len:].clone()  # (L, L)
            del A, attn_layers                            # free full-sequence tensor

            # ppmi_score(i) = Σ_{j≠i, j masked} A_resp[i,j] · confidence[j]
            j_mask   = (input_ids[0, prompt_len:] == mask_token_id).float()  # (L,)
            off_diag = 1.0 - torch.eye(L, device=device)                      # (L, L)
            # A_resp[i,j] * confidence[j] (broadcast) * j_mask[j] * off_diag[i,j]
            score = (
                A_resp
                * confidence.unsqueeze(0)   # (1, L) → (L, L)
                * j_mask.unsqueeze(0)       # zero out j positions already unmasked
                * off_diag                  # zero out diagonal (j == i)
            ).sum(dim=-1)                   # (L,)

            # Mirror the EOS suppression: also suppress i when i predicts EOS
            if cfg.zero_eos_confidence:
                score = score.masked_fill(eos_mask, 0.0)

            del A_resp
        else:
            score = confidence

        # -----------------------------------------------------------------
        # Lock already-unmasked positions: guarantee they are always selected
        # -----------------------------------------------------------------
        current_response = input_ids[0, prompt_len:]          # (L,)
        already_unmasked = (current_response != mask_token_id)
        score    = score.masked_fill(already_unmasked, float("inf"))
        pred_ids = torch.where(already_unmasked, current_response, pred_ids)

        # Number of tokens that should be unmasked at time s
        n_unmask = max(0, min(math.floor(L * (1.0 - s)), L))

        _, top_indices = torch.topk(score, k=n_unmask, largest=True)
        new_response = torch.full((L,), mask_token_id, dtype=torch.long, device=device)
        new_response[top_indices] = pred_ids[top_indices]

        input_ids[0, prompt_len:] = new_response

    # Drop everything after the first EOS token (if present)
    final = input_ids[0, prompt_len:]
    eos_positions = (final == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        final = final[: eos_positions[0]]

    return final

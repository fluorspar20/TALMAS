"""
Unified GSM8K evaluation CLI — supports all four scoring × suppression combinations:

  Scoring      TALMAS    Description
  ----------   -------   -------------------------------------------
  confidence   off       Vanilla LLaDA baseline (Algorithm 5)
  confidence   on        TALMAS attention suppression only
  ppmi         off       PPMI position scoring only
  ppmi         on        PPMI scoring + TALMAS attention suppression

Usage examples:

  # Vanilla LLaDA baseline
  python scripts/unified_gsm8k_eval.py --scoring confidence

  # PPMI scoring (no TALMAS)
  python scripts/unified_gsm8k_eval.py --scoring ppmi

  # TALMAS only (confidence scoring)
  python scripts/unified_gsm8k_eval.py --scoring confidence --talmas

  # PPMI + TALMAS
  python scripts/unified_gsm8k_eval.py --scoring ppmi --talmas --lambda-max 4.0 --mu 0.1

  # All flags
  python scripts/unified_gsm8k_eval.py \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --scoring ppmi --layer-agg last_half \\
      --talmas --lambda-max 4.0 --mu 0.1 \\
      --max_samples 100 --steps 128 \\
      --checkpoint ckpt_ppmi_talmas.jsonl \\
      --output-dir results

Requirements:
  pip install torch transformers datasets accelerate tqdm
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SamplingConfig, TALMASConfig, BASE_CONFIG, INSTRUCT_CONFIG
from src.utils import (
    build_prompt,
    extract_answer,
    answers_match,
    resolve_special_tokens,
    load_model_and_tokenizer,
)
from src.ppmi_sampling import flexible_remasking_sample, SDPACapture
from src.talmas import TALMASHookManager


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args) -> float:
    is_instruct = "instruct" in args.model.lower()
    cfg = INSTRUCT_CONFIG if is_instruct else BASE_CONFIG

    if args.generation_length:
        cfg.generation_length = args.generation_length
    if args.steps:
        cfg.steps = args.steps

    # TALMAS config
    talmas_cfg: Optional[TALMASConfig] = None
    if args.talmas:
        talmas_cfg = TALMASConfig(
            lambda_max=args.lambda_max,
            mu=args.mu,
            use_timestep_gate=not args.no_timestep_gate,
            use_layer_gate=not args.no_layer_gate,
        )

    # --- Startup info ---
    print(f"Model:             {args.model}")
    print(f"Mode:              {'Instruct' if is_instruct else 'Base'}")
    print(f"Scoring:           {args.scoring}"
          + (f"  (layer_agg={args.layer_agg})" if args.scoring == "ppmi" else ""))
    print(f"Generation length: {cfg.generation_length}")
    print(f"Sampling steps:    {cfg.steps}")
    print(f"Zero EOS conf:     {cfg.zero_eos_confidence}")
    print(f"Samples:           {args.max_samples or 'all'}")
    if talmas_cfg:
        print(f"TALMAS:            λ_max={talmas_cfg.lambda_max}  μ={talmas_cfg.mu}  "
              f"timestep_gate={talmas_cfg.use_timestep_gate}  "
              f"layer_gate={talmas_cfg.use_layer_gate}")
    else:
        print("TALMAS:            disabled")
    print()

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Eager attention is required for:
    #   - TALMAS  (F.sdpa monkey-patch must fire)
    #   - PPMI    (output_attentions=True is unsupported by Flash Attention 2)
    eager_attn = args.talmas or (args.scoring == "ppmi")
    tokenizer, model = load_model_and_tokenizer(args.model, eager_attn=eager_attn)

    mask_token_id, eos_token_id = resolve_special_tokens(tokenizer, model)
    print(f"mask_token_id={mask_token_id}, eos_token_id={eos_token_id}\n")

    # ------------------------------------------------------------------ #
    # Load dataset                                                         #
    # ------------------------------------------------------------------ #
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # ------------------------------------------------------------------ #
    # Resume from checkpoint if present                                    #
    # ------------------------------------------------------------------ #
    results: list = []
    correct = 0

    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        correct = sum(r["correct"] for r in results)
        n_done  = len(results)
        print(f"Resuming from checkpoint '{args.checkpoint}': "
              f"{n_done} examples already done ({correct}/{n_done} correct).")
        dataset = dataset.select(range(n_done, len(dataset)))

    total = len(results)
    print(f"Evaluating on {len(dataset)} remaining examples...\n")

    # ------------------------------------------------------------------ #
    # Set up PPMI capture (MUST come before TALMAS)                        #
    # TALMASHookManager captures F.sdpa as a closure at __init__ time;    #
    # installing SDPACapture first makes TALMAS's original_sdpa point     #
    # through the capture wrapper, so we get TALMAS-biased weights.       #
    # ------------------------------------------------------------------ #
    ppmi_capture = None
    if args.scoring == "ppmi":
        ppmi_capture = SDPACapture().install()

    # ------------------------------------------------------------------ #
    # Set up TALMAS hooks (after capture)                                  #
    # ------------------------------------------------------------------ #
    hook_manager = None
    if talmas_cfg is not None and talmas_cfg.lambda_max > 0.0:
        hook_manager = TALMASHookManager(model, talmas_cfg)

    # ------------------------------------------------------------------ #
    # Eval loop                                                            #
    # ------------------------------------------------------------------ #
    try:
        for example in tqdm(dataset, desc="GSM8K"):
            question  = example["question"]
            gold_full = example["answer"]
            gold_ans  = extract_answer(gold_full)

            prompt_text = build_prompt(question, is_instruct)
            prompt_ids  = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(device)

            output_ids = flexible_remasking_sample(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                cfg=cfg,
                device=device,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                hook_manager=hook_manager,
                scoring=args.scoring,
                layer_agg=args.layer_agg,
                sdpa_capture=ppmi_capture,
            )

            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            pred_ans    = extract_answer(output_text)
            is_correct  = answers_match(pred_ans, gold_ans)

            correct += int(is_correct)
            total   += 1

            entry = {
                "question":   question,
                "gold":       gold_ans,
                "prediction": pred_ans,
                "output":     output_text,
                "correct":    is_correct,
            }
            results.append(entry)

            # Append to checkpoint immediately so progress survives preemption
            if args.checkpoint:
                with open(args.checkpoint, "a") as ckpt_f:
                    ckpt_f.write(json.dumps(entry) + "\n")

            status = "✓" if is_correct else "✗"
            running_acc = correct / total * 100
            tqdm.write(
                f"[{total:>4}] {status}  gold={gold_ans:<8}  pred={pred_ans:<8}  "
                f"running acc: {correct}/{total} ({running_acc:.1f}%)"
            )
            if args.verbose:
                tqdm.write(f"       Q: {question[:80]}...")
                tqdm.write(f"       Output: {output_text[:200]}")
    finally:
        if hook_manager is not None:
            hook_manager.remove()
        if ppmi_capture is not None:
            ppmi_capture.uninstall()

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"GSM8K Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*50}")
    print(f"\nPaper reports: 70.3% (Base, 4-shot) / 69.4% (Instruct)")

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    out_path = _resolve_output_path(args)
    if out_path:
        payload = {
            "model":       args.model,
            "scoring":     args.scoring,
            "layer_agg":   args.layer_agg if args.scoring == "ppmi" else None,
            "accuracy":    accuracy,
            "correct":     correct,
            "total":       total,
            "sampling":    cfg.__dict__,
            "talmas":      talmas_cfg.__dict__ if talmas_cfg else None,
            "results":     results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return accuracy


def _resolve_output_path(args) -> Optional[str]:
    """Return the path to write the JSON results file, or None."""
    if args.output_file:
        return args.output_file

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        scoring = args.scoring                              # "confidence" | "ppmi"
        backend = "talmas" if args.talmas else "base"
        fname   = f"gsm8k_{scoring}_{backend}_{ts}.json"
        return os.path.join(args.output_dir, fname)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLaDA GSM8K evaluation — confidence vs PPMI scoring, with optional TALMAS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model / dataset ---
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Base",
                        help="HuggingFace model name or local path")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="GSM8K split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of examples (None = full 1319-example test set)")
    parser.add_argument("--generation_length", type=int, default=None,
                        help="Override generation length")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override number of diffusion steps")

    # --- Scoring ---
    scoring = parser.add_argument_group("Scoring options")
    scoring.add_argument("--scoring", type=str, default="confidence",
                         choices=["confidence", "ppmi"],
                         help="Unmasking-order scoring: max-softmax confidence (baseline) "
                              "or PPMI-inspired attention-coupling score")
    scoring.add_argument("--layer-agg", type=str, default="all",
                         choices=["all", "last_half"],
                         dest="layer_agg",
                         help="Which layers to average attention over for PPMI scoring: "
                              "'all' (every layer) or 'last_half' (upper half of the network)")

    # --- Output ---
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results to this specific JSON path (overrides --output-dir)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for auto-named results JSON; "
                             "filename is gsm8k_{scoring}_{base|talmas}_{timestamp}.json")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="JSONL file for incremental checkpointing; resumes from this "
                             "file if it already exists")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example predictions")

    # --- TALMAS ---
    talmas = parser.add_argument_group("TALMAS options")
    talmas.add_argument("--talmas", action="store_true",
                        help="Enable TALMAS attention suppression")
    talmas.add_argument("--lambda-max", type=float, default=4.0,
                        help="λ_max: maximum logit suppression magnitude")
    talmas.add_argument("--mu", type=float, default=0.1,
                        help="μ: mask→mask suppression scale (0=full, 1=same as real→mask)")
    talmas.add_argument("--no-timestep-gate", action="store_true",
                        help="Disable f(1-r_t) quadratic timestep gate")
    talmas.add_argument("--no-layer-gate", action="store_true",
                        help="Disable g(ℓ/L) sigmoid layer gate")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)

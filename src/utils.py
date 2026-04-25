"""
Shared utilities: prompting, answer extraction, token ID resolution, model loading.

Prompt and extraction logic is verbatim from llada_gsm8k_eval.py to guarantee
identical behaviour between the baseline script and the TALMAS evaluation path.
"""

import re
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# 4-shot GSM8K prompt  (verbatim from llada_gsm8k_eval.py)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """\
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

"""


def build_prompt(question: str, is_instruct: bool) -> str:
    """Build the 4-shot prompt for a given question."""
    if is_instruct:
        system = "Solve the following math problem step by step. At the end, state 'The answer is X.' where X is the numeric answer."
        shots = FEW_SHOT_EXAMPLES.strip()
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{shots}\n\nQuestion: {question}\nAnswer:"
            f"<|assistant|>\n"
        )
    else:
        return FEW_SHOT_EXAMPLES + f"Question: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Answer extraction  (verbatim from llada_gsm8k_eval.py)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer.
    Looks for 'The answer is X' or '#### X' patterns (GSM8K convention).
    """
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    m = re.search(r"[Tt]he answer is\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def answers_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(pred) == float(gold.replace(",", ""))
    except ValueError:
        return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Special token ID resolution  (extracted from llada_gsm8k_eval.py lines 257–290)
# ---------------------------------------------------------------------------

def resolve_special_tokens(tokenizer, model) -> Tuple[int, int]:
    """
    Return (mask_token_id, eos_token_id) for LLaDA.

    LLaDA uses a custom tokenizer that does NOT set the standard HuggingFace
    mask_token_id attribute.  Resolution order:
      1. tokenizer.mask_token_id  (standard HF attribute — usually None for LLaDA)
      2. convert "[MASK]" string  (some tokenizer configs register it this way)
      3. model.config.mask_token_id  (custom config field LLaDA sets)
      4. hardcoded fallback 126336  (known value from GSAI-ML/LLaDA-8B-* repos)
    """
    mask_token_id = tokenizer.mask_token_id

    if mask_token_id is None:
        encoded = tokenizer.convert_tokens_to_ids("[MASK]")
        if encoded != tokenizer.unk_token_id:
            mask_token_id = encoded

    if mask_token_id is None and hasattr(model, "config"):
        mask_token_id = getattr(model.config, "mask_token_id", None)

    if mask_token_id is None:
        mask_token_id = 126336
        print(
            f"WARNING: mask_token_id not found in tokenizer/config, "
            f"falling back to known LLaDA value: {mask_token_id}"
        )

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and hasattr(model, "config"):
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = 126081
        print(f"WARNING: eos_token_id not found, falling back to: {eos_token_id}")

    return mask_token_id, eos_token_id


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, eager_attn: bool = False):
    """
    Load LLaDA tokenizer and model.

    When eager_attn=True, forces attn_implementation="eager" so that the
    TALMAS hook (which monkey-patches F.scaled_dot_product_attention) fires
    correctly.  Flash Attention 2 bypasses F.sdpa and would silently skip the
    hook without this flag.
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model{'  [eager attention]' if eager_attn else ''}...")
    kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if eager_attn:
        kwargs["attn_implementation"] = "eager"

    model = AutoModel.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model

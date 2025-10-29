#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask the if-condition in an external testset and evaluate a trained model.

Input CSV columns: id, code, code_tokens, docstring, docstring_tokens
Output CSV: external-eval-results.csv with columns:
  id, input, is_correct, expected, predicted, score
"""

import os, re, ast, argparse
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

# ---------- helpers ----------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def token_f1(pred: str, gold: str) -> float:
    p = normalize_space(pred).split()
    g = normalize_space(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    prec = overlap / max(1, sum(cp.values()))
    rec  = overlap / max(1, sum(cg.values()))
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

def mask_one_if_condition_from_code(code: str):
    """
    AST-first: replace the test of the FIRST `if` with <mask_if>.
    Returns (masked_code, condition) or (None, None) if not possible.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return None, None

    if_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    if not if_nodes:
        return None, None

    node = if_nodes[0]  # deterministic
    # Try to recover the exact source slice for the test
    cond_src = None
    try:
        cond_src = ast.get_source_segment(code, node.test)
    except Exception:
        try:
            cond_src = ast.unparse(node.test)  # py>=3.9
        except Exception:
            pass
    if not cond_src:
        return None, None

    # Replace only on the line where "if ... :" begins
    lines = code.splitlines(keepends=True)
    ln = node.lineno - 1
    if not (0 <= ln < len(lines)):
        return None, None
    line = lines[ln]

    # robust, match up to colon on that line
    m = re.search(r"\bif\s*(.+?)\s*:\s*", line)
    if not m:
        return None, None
    s, e = m.span(1)
    lines[ln] = line[:s] + "<mask_if>" + line[e:]
    return "".join(lines), cond_src.strip()

def prepare_external_inputs(csv_path: str):
    df = pd.read_csv(csv_path)
    masked_inputs, golds, row_ids = [], [], []
    for _, r in df.iterrows():
        code = str(r["code"])
        masked, cond = mask_one_if_condition_from_code(code)
        if masked and cond:
            masked_inputs.append(masked)
            golds.append(cond)
            row_ids.append(r["id"])
    return row_ids, masked_inputs, golds

def ensure_specials(tokenizer, model):
    # make sure specials exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    # ensure mask_if and a generic <mask> exist
    add = []
    if "<mask_if>" not in (tokenizer.additional_special_tokens or []):
        add.append("<mask_if>")
    if tokenizer.mask_token is None:
        add.append("<mask>")
    if add:
        tokenizer.add_special_tokens({"additional_special_tokens": add})
    model.resize_token_embeddings(len(tokenizer))

    # sync config + generation_config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

def generate_predictions(model, tokenizer, inputs, batch_size=8, max_new_tokens=16):
    model.eval()
    preds = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"].to(model.device),
                attention_mask=enc["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                num_beams=1,                 # start greedy for stability
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        preds.extend(texts)
    return preds

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", help="Path to external testset CSV", default="python_extractor/benchmark_if_only.csv")
    ap.add_argument("--tok_dir", default="python_extractor/runs_dir_mlm/tokenizer")
    ap.add_argument("--model_dir", default="python_extractor/runs_dir_mlm/model")
    ap.add_argument("--out_csv", default="external-eval-results.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Prepare data
    row_ids, masked_inputs, golds = prepare_external_inputs(args.test_csv)
    print(f"Prepared {len(masked_inputs)} masked samples.")

    # Load tokenizer + model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tok_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(args.device)
    ensure_specials(tokenizer, model)

    # Predict
    preds = generate_predictions(
        model, tokenizer, masked_inputs,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens
    )

    # Score
    rows = []
    for rid, inp, gold, pred in zip(row_ids, masked_inputs, golds, preds):
        em = int(normalize_space(pred) == normalize_space(gold))
        f1 = token_f1(pred, gold)
        score = int(round(100 * max(em, f1)))
        rows.append({
            "id": rid,
            "input": inp,
            "is_correct": str(bool(em)).lower(),
            "expected": gold,
            "predicted": pred,
            "score": score
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()

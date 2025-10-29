import os
import torch

def pick_best_gpu_by_torch() -> int:
    """Select the GPU with highest available memory"""
    if not torch.cuda.is_available():
        return -1  # CPU fallback
    free_bytes = []
    for i in range(torch.cuda.device_count()):
        f_bytes, _ = torch.cuda.mem_get_info(i)  # (free, total) in bytes
        free_bytes.append((i, f_bytes))
    best = max(free_bytes, key=lambda x: x[1])[0]
    # After setting this, the selected GPU will appear as index 0 inside the process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
    print(f"[GPU PICK] Selected physical GPU {best} with largest free VRAM. "
          f"Inside this process it will be visible as cuda:0.")
    return best

pick_best_gpu_by_torch()

import os, json, random, re, argparse
from typing import List, Dict, Any, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    T5Config, T5ForConditionalGeneration, PreTrainedTokenizerFast,
    Trainer, TrainingArguments
)
from transformers.trainer_utils import set_seed
from torch.utils.data import Dataset
import torch
import ast
try:
    from ast import unparse as ast_unparse  # Py3.9+
except Exception:
    ast_unparse = None
try:
    import astor  # fallback for unparse on older Pythons
except Exception:
    astor = None


def ensure_dir(p: str):
    """Check if directory exists, else create it"""
    os.makedirs(p, exist_ok=True)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def safe_unparse(node: ast.AST) -> str:
    if ast_unparse:
        return ast_unparse(node)
    if astor is not None:
        return astor.to_source(node).rstrip()
    # last resort: rough string
    return str(node)


def train_tokenizer_from_functions_json(
    functions_json: str,
    out_dir: str,
    vocab_size: int = 32000,
    min_freq: int = 2
):
    ensure_dir(out_dir)

    # Gather function code to train tokenizer
    data = read_json(functions_json)
    tmp_corpus = os.path.join(out_dir, "corpus.txt")
    with open(tmp_corpus, "w", encoding="utf-8") as f:
        for row in data:
            code = row.get("function_code", "")
            if code:
                f.write(code.replace("\n", "\n") + "\n\n")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=[
            "<pad>", "<s>", "</s>", "<unk>", "<mask>", "<extra_id_0>", "<extra_id_1>",
            "<extra_id_2>", "<extra_id_3>", "<extra_id_4>", "<extra_id_5>",
            "<extra_id_6>", "<extra_id_7>", "<extra_id_8>", "<extra_id_9>",
            "<mask_if>"  # special token to replace if's in fine-tuning
        ],
    )
    tokenizer.train([tmp_corpus], trainer=trainer)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

    # Wrap as HF tokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(out_dir, "tokenizer.json"),
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<s>",
        unk_token="<unk>",
        mask_token="<mask>",
        additional_special_tokens=[f"<extra_id_{i}>" for i in range(10)] + ["<mask_if>"],
    )
    hf_tok.save_pretrained(out_dir)
    print(f"Tokenizer trained and saved to {out_dir}")
    return hf_tok


def t5_span_corrupt(text: str, noise_density=0.15, mean_span_len=3):
    """
    Returns (inputs, targets) for T5 denoising:
      inputs: original text with spans replaced by sentinel tokens <extra_id_k>
      targets: concatenation of removed spans in order, each prefixed by the same sentinel
    """
    tokens = list(text)
    num_noise = max(1, int(len(tokens) * noise_density))
    spans = []
    i = 0
    while i < num_noise:
        span_len = max(1, int(random.expovariate(1 / mean_span_len)))
        spans.append(span_len)
        i += span_len
    # choose start indices
    possible_starts = list(range(len(tokens)))
    random.shuffle(possible_starts)
    starts = sorted(possible_starts[:len(spans)])
    # build masks
    masked = [False] * len(tokens)
    for s, span_len in zip(starts, spans):
        for j in range(s, min(len(tokens), s + span_len)):
            masked[j] = True

    inputs, targets = [], []
    cur_mask = False
    sent_id = 0
    for idx, ch in enumerate(tokens):
        if masked[idx] and not cur_mask:
            inputs.append(f"<extra_id_{sent_id}>")
            cur_mask = True
        if not masked[idx] and cur_mask:
            targets.append(f"<extra_id_{sent_id}>")
            sent_id += 1
            cur_mask = False
        if masked[idx]:
            targets.append(ch)
        else:
            inputs.append(ch)
    if cur_mask:
        targets.append(f"<extra_id_{sent_id}>")

    return "".join(inputs), "".join(targets)


class PretrainDataset(Dataset):
    def __init__(self, rows: List[str], tokenizer: PreTrainedTokenizerFast, max_len=512):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text = self.rows[idx]
        inp, tgt = t5_span_corrupt(text)
        enc = self.tok(inp, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        dec = self.tok(tgt, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        labels = dec["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
    

class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=256, mask_prob=0.15):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)  # "<mask>"
        self.special_ids = set([tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id])

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        # mask 15% of non-special tokens
        prob = torch.full(labels.shape, self.mask_prob)
        maskable = ~torch.tensor([i.item() in self.special_ids for i in labels])
        to_mask = (torch.bernoulli(prob).bool()) & maskable

        # replace inputs with <mask>, keep labels as original at masked spots
        input_ids[to_mask] = self.mask_id
        # ignore loss where not masked
        labels[~to_mask] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,  # T5 trainer will handle shifting
        }


class IfMaskedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_token = tokenizer.mask_token or "<mask>"

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        code = self.pairs[i]["masked_function"]
        gold = self.pairs[i]["condition"]

        # k = # subword tokens in gold
        gold_ids = self.tok(gold, add_special_tokens=False)["input_ids"]
        k = max(1, len(gold_ids))

        # replace <mask_if> with k mask tokens
        masked = code.replace("<mask_if>", " ".join([self.mask_token]*k))

        enc = self.tok(masked, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        dec = self.tok(gold, truncation=True, max_length=min(128, k+8),
                       padding="max_length", return_tensors="pt")

        labels = dec["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }



def mask_one_if_condition(code: str) -> Optional[Dict[str, str]]:
    """Replaces exactly one if condition with <mask_if>"""
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    if_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    if not if_nodes:
        return None

    node = random.choice(if_nodes)
    try:
        cond_str = normalize_space(safe_unparse(node.test))
    except Exception:
        return None

    lines = code.splitlines(keepends=True)
    if hasattr(node, "lineno"):
        ln = node.lineno - 1
        line = lines[ln]
        m = re.search(r"\bif\s*(.+)\s*:\s*", line)
        if not m:
            return None
        start, end = m.span(1)
        masked_line = line[:start] + "<mask_if>" + line[end:]
        lines[ln] = masked_line
        masked_function = "".join(lines)
        return {"masked_function": masked_function, "condition": cond_str}
    return None


class IfFineTuneDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer: PreTrainedTokenizerFast, max_len=512):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ex = self.pairs[idx]
        enc = self.tok(ex["masked_function"], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        dec = self.tok(ex["condition"], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        labels = dec["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def load_functions(functions_json: str) -> List[str]:
    data = read_json(functions_json)
    codes = []
    for r in data:
        code = r.get("function_code", "")
        if not code:
            continue
        # Filter trivial/huge
        nlines = code.count("\n") + 1
        if 3 <= nlines <= 300:
            codes.append(code)
    random.shuffle(codes)
    return codes

def build_pretrain_rows(codes: List[str], limit: Optional[int] = None) -> List[str]:
    if limit:
        codes = codes[:limit]
    return codes

def build_finetune_pairs(codes: List[str], target_count=50000) -> List[Dict[str, str]]:
    pairs = []
    for code in codes:
        sample = mask_one_if_condition(code)
        if sample:
            # Keep targets reasonably short
            if len(sample["condition"]) <= 200:
                pairs.append(sample)
        if len(pairs) >= target_count:
            break
    return pairs


def build_t5_from_scratch(vocab_size: int, d_model=256, num_layers=4, num_heads=8, d_ff=1024):
    config = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=0.1,
        eos_token_id=1,  # </s> by our tokenizer config
        pad_token_id=0,  # <pad>
    )
    model = T5ForConditionalGeneration(config)
    return model


def simple_f1(pred: str, gold: str) -> float:
    # token-level F1 for conditions
    p = normalize_space(pred).split()
    g = normalize_space(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w)) if w in g else 0
    overlap = sum(common.values())
    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def generate_predictions(model, tokenizer, inputs: List[str], batch_size=8, max_new_tokens=64) -> List[str]:
    preds = []
    model.eval()
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"].to(model.device),
                attention_mask=enc["attention_mask"].to(model.device),
                max_new_tokens=32,             
                num_beams=4,                   
                early_stopping=True,
                no_repeat_ngram_size=3,        
                repetition_penalty=1.15,       
                decoder_start_token_id=tokenizer.pad_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(texts)
    return preds


def predict_k_tokens(model, tok, masked_funcs, golds, batch_size=8):
    preds = []
    model.eval()
    for b in range(0, len(masked_funcs), batch_size):
        batch = masked_funcs[b:b+batch_size]
        ks = [len(tok(g, add_special_tokens=False)["input_ids"]) for g in golds[b:b+batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"].to(model.device),
                attention_mask=enc["attention_mask"].to(model.device),
                max_new_tokens=max(ks),
                num_beams=1,                 
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                decoder_start_token_id=tok.pad_token_id,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # enforce exactly k tokens
        trimmed = []
        for t, k in zip(texts, ks):
            ids = tok(t, add_special_tokens=False)["input_ids"][:k]
            trimmed.append(tok.decode(ids, skip_special_tokens=True))
        preds.extend(trimmed)
    return preds


def streamline_tokenize(tokenizer, model):
    # ensure specials exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})

    # keep config/generation config in sync
    tokenizer.mask_token = "<mask>"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id  = tokenizer.eos_token_id

    # if tokens were added
    model.resize_token_embeddings(len(tokenizer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_json", type=str, required=True, help="Path to functions.json")
    parser.add_argument("--workdir", type=str, default="ai4se_out")
    parser.add_argument("--seed", type=int, default=42)

    # tokenizer
    parser.add_argument("--tok_dir", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_freq", type=int, default=2)

    # pretrain
    parser.add_argument("--do_pretrain", action="store_true")
    parser.add_argument("--pretrain_rows", type=int, default=150000)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--pretrain_bs", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=5e-4)

    # finetune
    parser.add_argument("--do_finetune", action="store_true")
    parser.add_argument("--finetune_rows", type=int, default=50000)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--finetune_bs", type=int, default=32)
    parser.add_argument("--finetune_lr", type=float, default=3e-4)

    # eval
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_rows", type=int, default=5000)
    parser.add_argument("--generated_csv", type=str, default="generated-testset.csv")

    args = parser.parse_args()
    set_seed(args.seed)

    ensure_dir(args.workdir)

    # --- Tokenizer ---
    tok_dir = args.tok_dir or os.path.join(args.workdir, "tokenizer")
    if not os.path.exists(tok_dir) or not os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        tokenizer = train_tokenizer_from_functions_json(
            functions_json=args.functions_json,
            out_dir=tok_dir,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq
        )
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
        print(f"Loaded existing tokenizer from {tok_dir}")

    # after loading/creating tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    # --- Model ---
    model_dir = os.path.join(args.workdir, "model")
    ensure_dir(model_dir)
    if os.listdir(model_dir):
        print("Loading existing model from", model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    else:
        print("Initializing T5 from scratch")
        model = build_t5_from_scratch(vocab_size=tokenizer.vocab_size)

    streamline_tokenize(tokenizer, model)

    # Put model on device via Trainer automatically

    # --- Load codes
    all_codes = load_functions(args.functions_json)

    # ----------------- PRE-TRAIN -----------------
    if args.do_pretrain:
        rows = build_pretrain_rows(all_codes, limit=args.pretrain_rows)
        split = int(0.95 * len(rows))
        train_rows = rows[:split]
        val_rows = rows[split:]

        # train_ds = PretrainDataset(train_rows, tokenizer)
        # val_ds = PretrainDataset(val_rows, tokenizer)

        train_ds = MLMDataset(train_rows, tokenizer, max_len=256, mask_prob=0.15)
        val_ds   = MLMDataset(val_rows, tokenizer, max_len=256, mask_prob=0.15)

        pretrain_out = os.path.join(args.workdir, "pretrain")
        ensure_dir(pretrain_out)

        targs = TrainingArguments(
            output_dir=pretrain_out,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.pretrain_bs,
            per_device_eval_batch_size=args.pretrain_bs,
            learning_rate=args.pretrain_lr,
            num_train_epochs=args.pretrain_epochs,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tok_dir)

    # ----------------- FINE-TUNE -----------------
    if args.do_finetune:
        pairs = build_finetune_pairs(all_codes[::-1], target_count=args.finetune_rows)
        random.shuffle(pairs)
        n = len(pairs)
        tr, va, te = int(0.8*n), int(0.9*n), n
        train_pairs, val_pairs, test_pairs = pairs[:tr], pairs[tr:va], pairs[va:te]

        # save generated testset JSONL for reproducibility
        gen_dir = os.path.join(args.workdir, "finetune_data")
        ensure_dir(gen_dir)
        write_jsonl(os.path.join(gen_dir, "train.jsonl"), train_pairs)
        write_jsonl(os.path.join(gen_dir, "val.jsonl"), val_pairs)
        write_jsonl(os.path.join(gen_dir, "test.jsonl"), test_pairs)

        # train_ds = IfFineTuneDataset(train_pairs, tokenizer)
        # val_ds = IfFineTuneDataset(val_pairs, tokenizer)
        train_ds = IfMaskedTokenDataset(train_pairs, tokenizer, max_len=512)
        val_ds = IfMaskedTokenDataset(val_pairs, tokenizer, max_len=512)

        finetune_out = os.path.join(args.workdir, "finetune")
        ensure_dir(finetune_out)
        targs = TrainingArguments(
            output_dir=finetune_out,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.finetune_bs,
            per_device_eval_batch_size=args.finetune_bs,
            learning_rate=args.finetune_lr,
            num_train_epochs=args.finetune_epochs,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=400,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tok_dir)

    # ----------------- EVAL (CSV) -----------------
    if args.do_eval:
        pairs = build_finetune_pairs(all_codes, target_count=max(2000, args.eval_rows))
        eval_pairs = pairs[:args.eval_rows]
        inputs = [p["masked_function"] for p in eval_pairs]
        golds  = [p["condition"] for p in eval_pairs]

        # preds = generate_predictions(model, tokenizer, inputs, batch_size=8)
        preds = predict_k_tokens(model=model, 
                                 tok=tokenizer, 
                                 masked_funcs=inputs, 
                                 golds=golds, 
                                 batch_size=8)
        rows = []
        for inp, gold, pred in zip(inputs, golds, preds):
            em = int(normalize_space(pred) == normalize_space(gold))
            f1 = simple_f1(pred, gold)
            score = int(round(100 * max(em, f1)))  # 0–100
            rows.append({
                "input": inp,
                "is_correct": str(bool(em)).lower(),
                "expected": gold,
                "predicted": pred,
                "score": score
            })

        out_csv = os.path.join(args.workdir, args.generated_csv)
        import csv
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["input", "is_correct", "expected", "predicted", "score"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote evaluation CSV to {out_csv}")


if __name__ == "__main__":
    main()

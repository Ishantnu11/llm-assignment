"""
Evaluate base model vs fine-tuned adapter on data/test.jsonl (held-out phrasings).

    python eval.py --tag base
    python eval.py --tag finetuned --adapter outputs/r16_a32_all

Metrics:
  valid_json   : output parses as JSON with exactly the 5 schema keys
  exact_match  : all 5 fields equal to gold  (main metric)
  field_acc    : per-field accuracy (amount, currency, category, merchant, date)
Writes results/<tag>_metrics.json and results/<tag>_predictions.jsonl (use this to pick failure cases).
"""
import argparse, json, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--adapter", default=None)
p.add_argument("--tag", required=True)
args = p.parse_args()

HERE = Path(__file__).parent
KEYS = ["amount", "currency", "category", "merchant", "date"]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)  # be lenient: grab first {...} block
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def same(field, pred, gold):
    if field == "amount":
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(pred, str) and isinstance(gold, str):
        return pred.strip().lower() == gold.strip().lower()
    return pred == gold


rows = [json.loads(l) for l in open(HERE / "data/test.jsonl", encoding="utf-8")]
counts = {"valid_json": 0, "exact_match": 0, **{k: 0 for k in KEYS}}
preds_out = []

for ex in rows:
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["input"]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    gold = json.loads(ex["output"])
    pred = parse(text)
    ok_fields = {}
    if pred is not None and set(pred) == set(KEYS):
        counts["valid_json"] += 1
    for k in KEYS:
        ok_fields[k] = pred is not None and same(k, pred.get(k), gold[k])
        counts[k] += ok_fields[k]
    if all(ok_fields.values()):
        counts["exact_match"] += 1
    preds_out.append({"input": ex["input"], "gold": gold, "raw_output": text,
                      "parsed": pred, "field_ok": ok_fields})

n = len(rows)
metrics = {"n": n, **{k: round(v / n, 4) for k, v in counts.items()}}
print(json.dumps(metrics, indent=2))

(HERE / "results").mkdir(exist_ok=True)
json.dump(metrics, open(HERE / "results" / f"{args.tag}_metrics.json", "w"), indent=2)
with open(HERE / "results" / f"{args.tag}_predictions.jsonl", "w", encoding="utf-8") as f:
    for r in preds_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

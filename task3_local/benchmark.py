"""
Benchmark one Ollama model tag on the 10 prompts: speed, memory, outputs.

    ollama pull qwen2.5:1.5b-instruct-fp16
    ollama pull qwen2.5:1.5b-instruct-q4_K_M
    python benchmark.py --model qwen2.5:1.5b-instruct-fp16
    python benchmark.py --model qwen2.5:1.5b-instruct-q4_K_M
    python compare.py   # side-by-side table + outputs for quality review

Speed numbers come straight from Ollama's own timers:
  decode tok/s  = eval_count / eval_duration          (generation speed)
  prefill tok/s = prompt_eval_count / prompt_eval_duration
Memory: model size reported by Ollama /api/ps (size and size_vram), sampled while loaded,
plus system RAM delta via psutil as a cross-check.
"""
import argparse, json, statistics, time
from pathlib import Path

import psutil
import requests

URL = "http://localhost:11434"
HERE = Path(__file__).parent

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--max_tokens", type=int, default=256)
args = p.parse_args()

prompts = json.load(open(HERE / "prompts.json", encoding="utf-8"))
opts = {"temperature": 0, "seed": 42, "num_predict": args.max_tokens}

# unload anything else, then measure RAM before load
requests.post(f"{URL}/api/generate", json={"model": args.model, "keep_alive": 0})
time.sleep(2)
ram_before = psutil.virtual_memory().used

# warm-up (loads weights; excluded from timing)
requests.post(f"{URL}/api/generate", json={"model": args.model, "prompt": "hi",
                                           "stream": False, "options": {"num_predict": 4}}, timeout=600)
ram_after = psutil.virtual_memory().used
ps = requests.get(f"{URL}/api/ps").json().get("models", [])
loaded = next((m for m in ps if m["name"].startswith(args.model)), {})

runs = []
for i, prompt in enumerate(prompts):
    r = requests.post(f"{URL}/api/generate", json={"model": args.model, "prompt": prompt,
                                                   "stream": False, "options": opts}, timeout=600).json()
    decode = r["eval_count"] / (r["eval_duration"] / 1e9)
    prefill = r.get("prompt_eval_count", 0) / max(r.get("prompt_eval_duration", 1) / 1e9, 1e-9)
    runs.append({"i": i, "prompt": prompt, "output": r["response"],
                 "out_tokens": r["eval_count"], "decode_tok_s": round(decode, 2),
                 "prefill_tok_s": round(prefill, 2)})
    print(f"[{i}] {decode:6.1f} tok/s  ({r['eval_count']} tokens)")

summary = {
    "model": args.model,
    "decode_tok_s_mean": round(statistics.mean(r["decode_tok_s"] for r in runs), 2),
    "decode_tok_s_median": round(statistics.median(r["decode_tok_s"] for r in runs), 2),
    "prefill_tok_s_mean": round(statistics.mean(r["prefill_tok_s"] for r in runs), 2),
    "ollama_model_size_GB": round(loaded.get("size", 0) / 1e9, 3),
    "ollama_size_vram_GB": round(loaded.get("size_vram", 0) / 1e9, 3),
    "system_ram_delta_GB": round((ram_after - ram_before) / 1e9, 3),
    "machine": {"cpu_count": psutil.cpu_count(), "ram_total_GB": round(psutil.virtual_memory().total / 1e9, 1)},
}
print(json.dumps(summary, indent=2))

(HERE / "results").mkdir(exist_ok=True)
safe = args.model.replace(":", "_").replace("/", "_")
json.dump({"summary": summary, "runs": runs},
          open(HERE / "results" / f"{safe}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

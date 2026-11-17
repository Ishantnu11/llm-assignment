"""
Reads every results/*.json from benchmark.py and writes results/comparison.md:
a speed/memory table plus each prompt's outputs side by side, with an empty
'my notes' line per prompt for the quality review (fill those in by hand).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
files = sorted(f for f in (HERE / "results").glob("*.json"))
data = [json.load(open(f, encoding="utf-8")) for f in files]
if not data:
    raise SystemExit("run benchmark.py first")

lines = ["# Speed & memory\n",
         "| Model | decode tok/s (mean) | prefill tok/s | Ollama size (GB) | VRAM (GB) | RAM delta (GB) |",
         "|---|---|---|---|---|---|"]
for d in data:
    s = d["summary"]
    lines.append(f"| {s['model']} | {s['decode_tok_s_mean']} | {s['prefill_tok_s_mean']} | "
                 f"{s['ollama_model_size_GB']} | {s['ollama_size_vram_GB']} | {s['system_ram_delta_GB']} |")

lines.append("\n# Quality: same 10 prompts\n")
for i in range(len(data[0]["runs"])):
    lines.append(f"## Prompt {i+1}\n> {data[0]['runs'][i]['prompt']}\n")
    for d in data:
        out = d["runs"][i]["output"].strip().replace("\n", "\n    ")
        lines.append(f"**{d['summary']['model']}**\n\n    {out}\n")
    lines.append("**My notes:** TODO (correct? same as other setting? what differs?)\n")

(HERE / "results" / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote results/comparison.md")

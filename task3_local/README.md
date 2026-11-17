# Task 3: Local inference and quantisation

**Runtime:** Ollama  **Model:** Qwen2.5-1.5B-Instruct in two settings: **FP16** vs **Q4_K_M** (4-bit)
**Machine:** TODO (CPU / GPU / RAM / OS)

## Run
```bash
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python benchmark.py --model qwen2.5:1.5b-instruct-fp16
python benchmark.py --model qwen2.5:1.5b-instruct-q4_K_M
python compare.py
```
Temperature 0 and a fixed seed, so the two settings can be compared fairly.

## Speed and memory (from `results/comparison.md`)
| Setting | decode tok/s | prefill tok/s | memory (GB) |
|---|---|---|---|
| FP16 | | | |
| Q4_K_M | | | |

## Quality on the same 10 prompts
Full outputs and per-prompt notes: `results/comparison.md`.
Summary of my observations: TODO

## Concepts (in your own words)
- **What the KV cache is:** TODO
- **Why quantisation saves memory:** TODO

## Serving 20 concurrent users on one 24 GB GPU
See `serving_plan.md`.

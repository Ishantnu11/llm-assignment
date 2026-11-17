# Back-of-envelope: 20 concurrent users on one 24 GB GPU

Fill in your own numbers and reasoning.

## Assumptions (state them clearly)
- Model + precision: ?
- Avg prompt length: ? tokens, avg response: ? tokens → context per user ≈ ?
- Engine: ? (e.g. vLLM with continuous batching + PagedAttention)
- Target: ? tok/s per user

## Memory budget
| Item | Working | GB |
|---|---|---|
| Model weights | params × bytes/param | ? |
| KV cache per token | 2 (K and V) × layers × kv_heads × head_dim × bytes | ? |
| KV cache per user | per-token × context length | ? |
| KV cache for 20 users | × 20 | ? |
| Runtime overhead / activations | assume ? | ? |
| **Total vs 24 GB** | | **?** |

(Get layers / kv_heads / head_dim from the model's `config.json` on Hugging Face.)

## Throughput check
Total tokens/s needed = 20 × target per-user tok/s = ? Is that realistic for this GPU, and why does batching help?

## What I'd do if it doesn't fit
TODO (e.g. smaller model, quantised weights, FP8 KV cache, shorter max context, a queue)

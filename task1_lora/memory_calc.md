# GPU memory to fine-tune a 7B model: worksheet

Fill in the numbers yourself and be ready to explain every line live.

**Given:** N = 7 × 10⁹ parameters. 1 GB ≈ 10⁹ bytes.

Memory has four parts: **weights + gradients + optimizer state + activations**.

## (a) Full fine-tuning (mixed precision, AdamW)
| Part | Bytes per param | Total (GB) |
|---|---|---|
| Weights (fp16/bf16) | ? | ? |
| Gradients (fp16/bf16) | ? | ? |
| fp32 master copy of weights | ? | ? |
| Adam 1st moment (fp32) | ? | ? |
| Adam 2nd moment (fp32) | ? | ? |
| Activations (depends on batch × seq len) | — | assume ? |
| **Total** | | **?** |

## (b) LoRA (base frozen in fp16, rank r adapters)
Trainable params ≈ ? (work it out: for each targeted matrix of shape d×k, LoRA adds r·(d+k) params; count layers × matrices).
| Part | Total (GB) |
|---|---|
| Frozen base weights (fp16) | ? |
| LoRA weights + grads + Adam state | ? |
| Activations | ? |
| **Total** | **?** |

## (c) QLoRA (base frozen in 4-bit NF4)
| Part | Total (GB) |
|---|---|
| Base weights at 4 bits (+ small quantisation-constant overhead) | ? |
| LoRA weights + grads + Adam state | ? |
| Activations (gradient checkpointing on) | ? |
| **Total** | **?** |

**Sanity check against my own run:** `results/*_log.json` records `peak_gpu_mem_GB` for the 0.5B QLoRA run. Scale that number up and see whether it roughly matches (c).

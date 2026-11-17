# Task 1: QLoRA fine-tuning (expense note → JSON)

**Task:** turn a messy expense note (`"ugh, Uber just took ₹340 for a cab yesterday"`) into a fixed JSON object with 5 keys.
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (0.5B params)
**Data:** 1200 train / 100 val / 100 test. The test set uses **phrasings never seen in training**.

## Run (Colab/Kaggle T4)
```bash
pip install -r ../requirements.txt
python data/make_dataset.py
python eval.py --tag base                                  # base model score
python train.py --run_name r16_a32_all                     # fine-tune
python eval.py --tag finetuned --adapter outputs/r16_a32_all
```
Optional ablation: `python train.py --rank 4 --alpha 8 --targets q_proj,v_proj --run_name r4_qv`, then eval it.

## Results (fill in from `results/*_metrics.json`)
| Model | valid JSON | exact match | amount | currency | category | merchant | date |
|---|---|---|---|---|---|---|---|
| Base | | | | | | | |
| Fine-tuned | | | | | | | |

## LoRA settings and why I chose them
| Setting | Value | Why (my reasoning) |
|---|---|---|
| rank | 16 | TODO |
| alpha | 32 | TODO |
| target modules | all attention + MLP projections | TODO |
| learning rate | 2e-4 | TODO |
| epochs | 3 | TODO |
| quantisation | 4-bit NF4 (QLoRA) | TODO |

## 3 failure cases (pick them from `results/finetuned_predictions.jsonl` where `field_ok` has a false)
1. Input / expected / got / why I think it fails: TODO
2. TODO
3. TODO

## GPU memory for fine-tuning a 7B model
See `memory_calc.md`.

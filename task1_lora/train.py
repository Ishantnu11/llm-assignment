"""
QLoRA fine-tune of a small instruct model on data/train.jsonl.

    python train.py                       # defaults below
    python train.py --rank 8 --epochs 2   # try other settings, compare in results/

Saves adapter to outputs/<run_name>/ and the training log to results/<run_name>_log.json
"""
import argparse, json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
p.add_argument("--run_name", default="r16_a32_all")
args = p.parse_args()

HERE = Path(__file__).parent
tok = AutoTokenizer.from_pretrained(args.model)

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
    target_modules=args.targets.split(","), task_type="CAUSAL_LM",
)


def to_messages(ex):
    return {"messages": [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]}


ds = load_dataset("json", data_files={"train": str(HERE / "data/train.jsonl"),
                                      "val": str(HERE / "data/val.jsonl")})
ds = ds.map(to_messages, remove_columns=["system", "input", "output"])

cfg = SFTConfig(
    output_dir=str(HERE / "outputs" / args.run_name),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=2,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    max_seq_length=256,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                     eval_dataset=ds["val"], peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(str(HERE / "outputs" / args.run_name))

(HERE / "results").mkdir(exist_ok=True)
with open(HERE / "results" / f"{args.run_name}_log.json", "w") as f:
    json.dump({"args": vars(args), "log_history": trainer.state.log_history,
               "peak_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9}, f, indent=2)
print("peak GPU mem (GB):", torch.cuda.max_memory_allocated() / 1e9)

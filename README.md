# Financial QLoRA Fine-tuning (Windows + CUDA)

This project fine-tunes a 3B model on Indian market data using QLoRA (4-bit) with PEFT/TRL.

## Quickstart (Windows PowerShell)

1. Create venv and install deps

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# Install CUDA-enabled PyTorch (adjust cu124 if needed)
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio --upgrade
# Install bitsandbytes (Windows wheel)
pip install bitsandbytes==0.43.3
```

2. Sanity check (no training, tiny model)

```powershell
python run.py --quick
```

3. Short GPU QLoRA validation run with validation monitoring

```powershell
# 4GB VRAM-friendly settings with 80/20 train/validation split
python run.py --epochs 1 --batch-size 1 --grad-accum 8 --lr 2e-4 --max-steps 50 --max-seq-len 512 --save-strategy steps --save-steps 25
```

**What's New:**

- **Chat template format**: Dataset now uses conversational messages format (system/user/assistant roles)
- **Expanded dataset**: 200+ samples with individual stocks (TCS, Reliance, HDFC, Infosys, ICICI)
- **Validation monitoring**: Automatic 80/20 train/validation split, evaluation every 25 steps
- **Early stopping**: Training stops automatically if validation loss doesn't improve for 3 evaluation cycles
- **Better quality**: Prevents overfitting that caused repetitive "###" outputs in previous version

4. Full fine-tune

```powershell
python run.py --epochs 3 --batch-size 1 --grad-accum 16 --lr 2e-4 --max-seq-len 512
```

Output adapters (LoRA) and tokenizer are saved under `./financial-qlora-model-final` after training.

## Tips

- VRAM limits (4GB): use `--batch-size 1`, increase `--grad-accum`, keep `--max-seq-len 512`, and ensure gradient checkpointing remains on (default in script for QLoRA).
- Optimizer: the script will use `paged_adamw_8bit` when `bitsandbytes` is present, otherwise fallback to `adamw_torch`.
- Fallbacks: if the base model fails to load, the script automatically falls back to a tiny model and skips training (`--quick`) for pipeline validation.

## Arguments

- `--epochs` (int): number of training epochs
- `--batch-size` (int): per-device train batch size
- `--grad-accum` (int): gradient accumulation steps
- `--lr` (float): learning rate (LoRA often uses 2e-4)
- `--max-steps` (int): overrides epochs if > 0
- `--max-seq-len` (int): tokenized max sequence length, default 512
- `--save-strategy` (epoch|steps): how to save checkpoints (default epoch)
- `--save-steps` (int): save every N steps if `--save-strategy steps` is set
- `--model` (str): override base model id (default stabilityai/stablelm-3b-4e1t)
- `--quick`: tiny-model pipeline check without training

## Troubleshooting

- OOM (OutOfMemory): lower `--max-seq-len` to 256–512, keep `--batch-size 1`, increase `--grad-accum`, ensure bitsandbytes is installed.
- Slow downloads: HF Hub may fall back to HTTP; consider installing `huggingface_hub[hf_xet]` for faster large file downloads.
- No CUDA: reinstall PyTorch with CUDA wheels for your driver version (e.g., cu124), or run `--quick` on CPU just to validate the pipeline.

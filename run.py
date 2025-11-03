# Financial Market QLoRA Fine-tuning Script
# Fine-tune a 3B model on Indian market data

# ============================================================================
# SECTION 1: Install Required Packages
# ============================================================================
# !pip install -q transformers accelerate peft bitsandbytes datasets trl
# !pip install -q yfinance pandas numpy torch

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

# ============================================================================
# SECTION 2: TRL imports (use correct modules)
# ============================================================================
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from trl.models.utils import setup_chat_format

# ============================================================================
# SECTION 3: Fetch Market Data (Past 100 Days)
# ============================================================================
print("Fetching market data...")

# Define Indian market tickers (verified working symbols)
tickers = {
    'NIFTY_50': '^NSEI',
    'SENSEX': '^BSESN',
    'NIFTY_MIDCAP': 'NIFTY_MIDCAP_100.NS',
    'NIFTY_SMALLCAP': '^CNXSC',  # CNX Smallcap index
    'GOLD_ETF': 'GOLDBEES.NS',
}

# Additional tickers for large cap, flexicap, and individual stocks
additional_tickers = {
    'LARGE_CAP': '^NSEBANK',  # Using Bank Nifty as large cap proxy
    'FLEXI_CAP': '0P0000XVH9.BO',  # Representative flexicap fund
    'TCS': 'TCS.NS',  # Individual stocks
    'RELIANCE': 'RELIANCE.NS',
    'HDFC_BANK': 'HDFCBANK.NS',
    'INFOSYS': 'INFY.NS',
    'ICICI_BANK': 'ICICIBANK.NS',
}

# Merge all tickers
all_tickers = {**tickers, **additional_tickers}

# Fetch data for past 100 days
end_date = datetime.now()
start_date = end_date - timedelta(days=140)  # Extra buffer for trading days

market_data = {}
for name, ticker in all_tickers.items():
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            market_data[name] = data.tail(100)  # Keep last 100 days
            print(f"[OK] Fetched {name}: {len(market_data[name])} days")
        else:
            print(f"[WARN] No data for {name}")
    except Exception as e:
        print(f"[ERROR] Error fetching {name}: {e}")

# ============================================================================
# SECTION 3: Prepare Training Dataset
# ============================================================================
print("\nPreparing training dataset...")

def create_training_samples(market_data):
    """Generate conversational instruction-response pairs in messages format"""
    samples = []
    
    # System message for financial analysis
    system_msg = "You are an expert financial analyst specializing in the Indian stock market. Provide accurate, concise analysis based on market data."

    for name, df in market_data.items():
        if len(df) < 5:
            continue

        # Convert to clean format
        df = df.copy()
        df['Date'] = df.index.strftime('%Y-%m-%d')
        ticker_name = name.replace('_', ' ')

        # Helper to extract scalar
        def get_val(x):
            return x.item() if hasattr(x, 'item') else float(x)

        # === PRICE QUERIES (multiple variations) ===
        latest = df.iloc[-1]
        close_price = get_val(latest['Close'])
        high_price = get_val(latest['High'])
        low_price = get_val(latest['Low'])
        volume = get_val(latest['Volume'])
        
        # Variation 1: Direct price
        samples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"What is the current price of {ticker_name}?"},
                {"role": "assistant", "content": f"The latest closing price of {ticker_name} is ₹{close_price:.2f} as of {latest['Date']}."}
            ]
        })
        
        # Variation 2: With intraday range
        samples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Tell me about {ticker_name}'s latest trading session."},
                {"role": "assistant", "content": f"In the latest session on {latest['Date']}, {ticker_name} closed at ₹{close_price:.2f}. The day's high was ₹{high_price:.2f} and low was ₹{low_price:.2f}, with {volume:,.0f} shares traded."}
            ]
        })
        
        # Variation 3: Volume focus
        samples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"What was the trading volume for {ticker_name} recently?"},
                {"role": "assistant", "content": f"On {latest['Date']}, {ticker_name} had a trading volume of {volume:,.0f} shares, with the price closing at ₹{close_price:.2f}."}
            ]
        })

        # === PERFORMANCE ANALYSIS (multiple timeframes) ===
        for days in [3, 5, 7]:
            if len(df) >= days:
                recent = df.tail(days)
                first_close = get_val(recent['Close'].iloc[0])
                last_close = get_val(recent['Close'].iloc[-1])
                price_change = ((last_close - first_close) / first_close) * 100
                direction = "gained" if price_change > 0 else "declined"
                
                samples.append({
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"How has {ticker_name} performed over the last {days} days?"},
                        {"role": "assistant", "content": f"Over the last {days} trading days, {ticker_name} {direction} {abs(price_change):.2f}%, moving from ₹{first_close:.2f} to ₹{last_close:.2f}."}
                    ]
                })

        # === VOLATILITY & RISK ===
        if len(df) >= 20:
            recent_20 = df.tail(20)
            volatility = get_val(recent_20['Close'].std())
            mean_close = get_val(recent_20['Close'].mean())
            max_close = get_val(recent_20['Close'].max())
            min_close = get_val(recent_20['Close'].min())
            vol_level = "high" if volatility > mean_close * 0.02 else "moderate"
            
            samples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"What is the volatility of {ticker_name}?"},
                    {"role": "assistant", "content": f"Based on 20-day analysis, {ticker_name} shows {vol_level} volatility with a standard deviation of ₹{volatility:.2f}. Price ranged from ₹{min_close:.2f} to ₹{max_close:.2f}."}
                ]
            })
            
            # Risk assessment
            samples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Is {ticker_name} a risky investment?"},
                    {"role": "assistant", "content": f"Based on recent 20-day volatility of ₹{volatility:.2f} ({vol_level} level), {ticker_name} shows {'moderate to higher' if vol_level == 'high' else 'relatively stable'} risk. Price has fluctuated between ₹{min_close:.2f} and ₹{max_close:.2f}."}
                ]
            })

        # === TREND ANALYSIS ===
        if len(df) >= 10:
            recent_10 = df.tail(10)
            closes = recent_10['Close'].values
            # Simple moving average trend
            sma_5 = closes[-5:].mean()
            latest_price = get_val(closes[-1])
            trend = "upward" if latest_price > sma_5 else "downward"
            
            samples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"What is the current trend for {ticker_name}?"},
                    {"role": "assistant", "content": f"The 10-day trend for {ticker_name} is {trend}, with the current price of ₹{latest_price:.2f} {'above' if trend == 'upward' else 'below'} the recent 5-day average of ₹{sma_5:.2f}."}
                ]
            })

    # === COMPARATIVE ANALYSIS ===
    if len(market_data) >= 2:
        names = list(market_data.keys())
        for i in range(min(len(names)-1, 5)):  # More comparisons
            for j in range(i+1, min(i+3, len(names))):  # Compare with next 2
                name1, name2 = names[i], names[j]
                df1, df2 = market_data[name1], market_data[name2]
                
                if len(df1) >= 10 and len(df2) >= 10:
                    for days in [5, 10]:
                        if len(df1) >= days and len(df2) >= days:
                            close1_first = get_val(df1['Close'].iloc[-days])
                            close1_last = get_val(df1['Close'].iloc[-1])
                            close2_first = get_val(df2['Close'].iloc[-days])
                            close2_last = get_val(df2['Close'].iloc[-1])
                            
                            ret1 = ((close1_last - close1_first) / close1_first) * 100
                            ret2 = ((close2_last - close2_first) / close2_first) * 100
                            better = name1.replace('_', ' ') if ret1 > ret2 else name2.replace('_', ' ')
                            
                            samples.append({
                                "messages": [
                                    {"role": "system", "content": system_msg},
                                    {"role": "user", "content": f"Compare {name1.replace('_', ' ')} vs {name2.replace('_', ' ')} over {days} days."},
                                    {"role": "assistant", "content": f"Over {days} days, {name1.replace('_', ' ')} returned {ret1:.2f}% while {name2.replace('_', ' ')} returned {ret2:.2f}%. {better} outperformed by {abs(ret1-ret2):.2f} percentage points."}
                                ]
                            })

    return samples

# Generate training samples
training_samples = create_training_samples(market_data)
print(f"Generated {len(training_samples)} training samples")

# Fallback sample to avoid empty dataset crashes if live data is unavailable
if len(training_samples) == 0:
    print("[WARN] No training samples generated from market data; adding a small fallback sample for correctness checks.")
    training_samples = [
        {
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": "What is the latest closing price of NIFTY 50?"},
                {"role": "assistant", "content": "I don't have current market data available. Please provide the latest data for analysis."}
            ]
        }
    ]

# === TRAIN/VALIDATION SPLIT ===
from sklearn.model_selection import train_test_split

print(f"\n[STATS] Dataset Statistics:")
print(f"  Total samples: {len(training_samples)}")

# Split into 80% train, 20% validation
train_samples, val_samples = train_test_split(
    training_samples, 
    test_size=0.2, 
    random_state=42
)

print(f"  Train samples: {len(train_samples)}")
print(f"  Validation samples: {len(val_samples)}")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_list(train_samples)
eval_dataset = Dataset.from_list(val_samples)

print(f"\n[OK] Created train and validation datasets")

# ============================================================================
# SECTION 4: Argument Parsing
# ============================================================================
print("\nConfiguring model...")

# Parse CLI args / quick mode for fast, correctness-only checks
parser = argparse.ArgumentParser(description="Financial Market QLoRA Fine-tuning")
parser.add_argument("--quick", action="store_true", help="Run a fast pipeline check with a tiny model and no training")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size (default: 4)")
parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
parser.add_argument("--max-steps", type=int, default=-1, help="Optional max training steps; overrides epochs if > 0")
parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for SFT tokenization (default: 512)")
parser.add_argument("--save-strategy", type=str, default="epoch", choices=["epoch", "steps"], help="Save strategy (default: epoch)")
parser.add_argument("--save-steps", type=int, default=50, help="Save every N steps when save-strategy=steps (default: 50)")
parser.add_argument("--model", type=str, default=None, help="Override base model id (e.g., meta-llama/Llama-3.2-3B)")
args, _ = parser.parse_known_args()
quick_mode = args.quick or os.getenv("QUICK_TEST", "0") == "1"
if quick_mode:
    print("⚡ Quick mode enabled: using a tiny model and skipping training for a fast correctness check.")


def detect_bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False

# Model selection
base_model_name = args.model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small instruct model with chat template
tiny_model_name = "sshleifer/tiny-gpt2"  # tiny CPU-friendly model
model_name = tiny_model_name if quick_mode else base_model_name

use_cuda = torch.cuda.is_available()
has_bnb = detect_bitsandbytes_available()
use_4bit = (not quick_mode) and use_cuda and has_bnb

quantization_config = None
if use_4bit:
    # Quantization config for QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# Load model with fallback to tiny model if loading fails
def load_model_and_tokenizer(name: str, quant_cfg):
    mdl = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quant_cfg,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return mdl, tok

try:
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config)
except Exception as e:
    print(f"[ERROR] Failed to load model '{model_name}' with error: {e}")
    print("↪ Falling back to tiny model without quantization for correctness.")
    model, tokenizer = load_model_and_tokenizer(tiny_model_name, None)
    quick_mode = True  # enforce quick mode behaviors on failure

# Prepare model for k-bit training only if using 4-bit
if use_4bit:
    model = prepare_model_for_kbit_training(model)
    # Enable memory optimizations
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# Ensure the tokenizer has a chat template; if not, set one up
try:
    if getattr(tokenizer, "chat_template", None) is None:
        # Use a standard chat format; chatml is broadly compatible
        model, tokenizer = setup_chat_format(
            model,
            tokenizer,
            format="chatml",
            resize_to_multiple_of=64,
        )
except Exception as e:
    print(f"[WARN] Failed to setup chat format automatically: {e}. Proceeding without explicit chat template.")

# ============================================================================
# SECTION 5: LoRA Configuration
# ============================================================================
print("Setting up LoRA...")

if quick_mode:
    print("Quick mode: Skipping LoRA setup.")
    lora_config = None
else:
    # Target a broad set of common projection layers across architectures
    lora_target_modules = [
        # Attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query_key_value",
        # MLP
        "gate_proj", "up_proj", "down_proj",
        "dense", "dense_h_to_4h", "dense_4h_to_h",
    ]
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ============================================================================
# SECTION 6: Training Configuration
# ============================================================================
print("Configuring training...")

if quick_mode:
    trainer = None
else:
    # Optimizer choice based on bitsandbytes availability
    optim_name = "paged_adamw_8bit" if has_bnb else "adamw_torch"

    sft_config = SFTConfig(
        output_dir="./financial-qlora-model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  # Eval batch size
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=use_cuda,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        logging_steps=10,
        optim=optim_name,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        # === VALIDATION MONITORING ===
        eval_strategy="steps",  # Evaluate every N steps
        eval_steps=25,  # Evaluate every 25 steps
        load_best_model_at_end=True,  # Load best checkpoint at end
        metric_for_best_model="eval_loss",  # Use validation loss
        greater_is_better=False,  # Lower loss is better
    )
    if args.max_steps and args.max_steps > 0:
        sft_config.max_steps = args.max_steps

    # === EARLY STOPPING CALLBACK ===
    from transformers import EarlyStoppingCallback
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement for 3 eval cycles
        early_stopping_threshold=0.01  # Minimum change to count as improvement
    )

    # Initialize trainer with validation dataset and callback
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,  # Use split train dataset
        eval_dataset=eval_dataset,  # Add validation dataset
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[early_stopping],  # Add early stopping
    )

# ============================================================================
# SECTION 7: Fine-tune the Model
# ============================================================================
if quick_mode:
    print("\nSkipping fine-tuning in quick mode.")
else:
    print("\n" + "="*60)
    print("STARTING TRAINING WITH VALIDATION MONITORING")
    print("="*60)
    print(f"[STATS] Train samples: {len(train_dataset)}")
    print(f"[STATS] Validation samples: {len(eval_dataset)}")
    print(f"[CONFIG] Evaluation every {sft_config.eval_steps} steps")
    print(f"[CONFIG] Early stopping patience: 3 evaluations")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print("="*60)

# ============================================================================
# SECTION 8: Save the Model
# ============================================================================
if quick_mode:
    print("\nQuick mode: Skipping model save.")
else:
    print("\nSaving model...")

    # Save LoRA adapters
    trainer.model.save_pretrained("./financial-qlora-model-final")
    tokenizer.save_pretrained("./financial-qlora-model-final")

    print("[OK] Model saved to ./financial-qlora-model-final")

# ============================================================================
# SECTION 9: Test the Fine-tuned Model
# ============================================================================
print("\nTesting model...")

def generate_response(instruction, context=""):
    # Build chat messages and apply chat template
    messages = [
        {"role": "system", "content": "You are an expert financial analyst for Indian markets."},
        {"role": "user", "content": f"{instruction}\n\nContext: {context}"},
    ]

    # Use tokenizer's chat template for generation prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=100 if quick_mode else 200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode only the generated continuation beyond the prompt
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = gen_out[0][input_len:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return decoded.strip()

# Build grounded contexts from fetched market_data for more accurate answers

def latest_context_for(ticker_key: str) -> str:
    if ticker_key not in market_data or len(market_data[ticker_key]) == 0:
        return f"No live data available for {ticker_key}."
    df = market_data[ticker_key]
    row = df.iloc[-1]
    date = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
    def gv(x):
        return x.item() if hasattr(x, 'item') else float(x)
    return (
        f"date={date}; close={gv(row['Close']):.2f}; high={gv(row['High']):.2f}; "
        f"low={gv(row['Low']):.2f}; volume={gv(row['Volume']):.0f}"
    )

# Test queries with grounded contexts
test_queries = [
    ("What is the latest price of NIFTY 50?", latest_context_for('NIFTY_50')),
    ("How has SENSEX performed recently? Give a 5-day summary.", latest_context_for('SENSEX')),
    ("Compare GOLD ETF with NIFTY MIDCAP over the most recent day.",
     f"GOLD_ETF: {latest_context_for('GOLD_ETF')} | NIFTY_MIDCAP: {latest_context_for('NIFTY_MIDCAP')}")
]

print("\n" + "="*60)
print("SAMPLE RESPONSES FROM MODEL")
print("="*60)

for instruction, context in test_queries:
    print(f"\n[QUERY] Query: {instruction}")
    print(f"Context: {context}")
    print(f"Response: {generate_response(instruction, context)}")
    print("-"*60)

if quick_mode:
    print("\n✅ Quick correctness check complete!")
else:
    print("\n✅ Fine-tuning and testing complete!")
    print("\nYour model is ready at: ./financial-qlora-model-final")
    print("\nTo use it later, load with:")
    print("from peft import PeftModel")
    print("base_model = AutoModelForCausalLM.from_pretrained('stabilityai/stablelm-3b-4e1t')")
    print("model = PeftModel.from_pretrained(base_model, './financial-qlora-model-final')")
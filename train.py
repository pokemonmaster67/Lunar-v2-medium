import os
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from model.config import LunarConfig
from model.transformer import LunarModel
from accelerate import Accelerator
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--use_150m", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize wandb
    wandb.init(project="lunar-v2-medium", config=args)
    
    # Load config and model
    if args.use_150m:
        config = LunarConfig.get_150m_config()
    else:
        config = LunarConfig()
    
    model = LunarModel(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_sequence_length)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    model.train()
    total_steps = 0
    
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                
                loss = outputs["loss"]
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                optimizer.zero_grad()
                
                total_steps += 1
                
                if total_steps % 100 == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "step": total_steps,
                        "epoch": epoch,
                    })
        
        # Save checkpoint
        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config,
                },
                os.path.join(args.output_dir, f"lunar_v2_medium_epoch_{epoch}.pt")
            )

if __name__ == "__main__":
    main()

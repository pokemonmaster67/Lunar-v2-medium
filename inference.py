import torch
import argparse
from transformers import AutoTokenizer
from model.config import LunarConfig
from model.transformer import LunarModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()

def generate(model, tokenizer, prompt, max_length, temperature=0.7, top_p=0.9):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token[0][0] == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    args = parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    config = checkpoint["config"]
    
    # Initialize model and tokenizer
    model = LunarModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Generate text
    generated_text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_length,
        args.temperature,
        args.top_p
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()

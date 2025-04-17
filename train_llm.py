import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gpt_2_Pytorch.GPT2.config import GPT2Config
from gpt_2_Pytorch.GPT2.model import GPT2LMHeadModel
from gpt_2_Pytorch.GPT2.utils import load_weight
from dataset import AlpacaGPT2Dataset, collate_fn
from tqdm import tqdm
from modify_llm import (
    apply_lora_to_model, 
    apply_adapters_to_model, 
    apply_prefix_layers_to_model,
    LoRAInjectionConv1D,
    LoRAInjection,
    PrefixInjection,
    AdapterInjection
)

##################################################### T R A I N   F U N C T I O N ###################################################

def train_gpt2(
    model,
    train_dataset,
    num_epochs=3,
    batch_size=2,
    learning_rate=1e-4,
    num_dataloader_workers=8,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
        collate_fn=collate_fn
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            loss = model(input_ids=input_ids, lm_labels=labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}")



##################################################### A R G U M E N T S   P A R S I N G ###################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-Tuning Script for GPT2 with LoRA, Prefix Layers, and Adapters"
    )

    parser.add_argument(
        "--finetuning_technique",
        type=str,
        required=True,
        choices=["lora", "adapters", "prefix"],
        help="Type of modification to apply to GPT2 model."
    )

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=float, default=32, help="Scaling factor for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout for LoRA layers.")

    # Adapter arguments
    parser.add_argument("--adapter_size", type=int, default=64, help="Hidden dimension in adapters.")
    parser.add_argument("--adapter_dropout", type=float, default=0.0, help="Dropout for adapter layers.")

    # Prefix arguments
    parser.add_argument("--prefix_size", type=float, default=1.0, help="Scaling factor for naive prefix layer.")
    parser.add_argument("--prefix_dropout", type=float, default=0.0, help="Dropout for prefix layer if desired.")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--num_dataloader_workers", type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g., 'cuda').")


    return parser.parse_args()



##################################################### M A I N  ##############################################################
if __name__ == "__main__":

    args = parse_arguments()

    # Load the model
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load(
            'gpt2-pytorch_model.bin',
            map_location='cpu' if not torch.cuda.is_available() else None
        )
        model = load_weight(model, state_dict)
        print("Model weights loaded successfully.")
    else:
        print('Weights gpt2-pytorch_model.bin not found')
        sys.exit()
    
    model.set_tied()

    # Apply the specified finetuning technique
    if args.finetuning_technique == "lora":
        print("Applying LoRA modifications to the GPT2 model...")
        model = apply_lora_to_model(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )
    elif args.finetuning_technique == "adapters":
        print("Applying Adapter layers to the GPT2 model...")
        model = apply_adapters_to_model(
            model,
            adapter_size=args.adapter_size,
            dropout=args.adapter_dropout
        )
    elif args.finetuning_technique == "prefix":
        print("Applying Prefix layers to the GPT2 model...")
        model = apply_prefix_layers_to_model(
            model,
            prefix_size=args.prefix_size,
            dropout=args.prefix_dropout
        )

    # Freeze all parameters except for the ones we want to train
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters based on the finetuning technique
    if args.finetuning_technique == "lora":
        for name, module in model.named_modules():
            if isinstance(module, LoRAInjection) or isinstance(module, LoRAInjectionConv1D):
                for pname, param in module.named_parameters():
                    if "lora_A" in pname or "lora_B" in pname:
                        param.requires_grad = True

    elif args.finetuning_technique == "adapters":
        for module in model.modules():
            if isinstance(module, AdapterInjection):
                module.adapter.requires_grad_(True)
                for param in module.adapter.parameters():
                    param.requires_grad = True


    elif args.finetuning_technique == "prefix":
        for module in model.modules():
            if isinstance(module, PrefixInjection):
                module.prefix.requires_grad_(True) 
                for pname, param in module.named_parameters():
                    if pname != "prefix":         
                        param.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    train_dataset = AlpacaGPT2Dataset(from_csv=True)

    # Train 
    train_gpt2(
        model,
        train_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_dataloader_workers=args.num_dataloader_workers,
        device=args.device
    )


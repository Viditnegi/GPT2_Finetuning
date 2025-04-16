import os
import sys
import argparse
import torch
import torch.nn as nn
from gpt_2_Pytorch.GPT2.config import GPT2Config
from gpt_2_Pytorch.GPT2.model import GPT2LMHeadModel
from gpt_2_Pytorch.GPT2.model import Conv1D
from gpt_2_Pytorch.GPT2.utils import load_weight


class LoRAInjection(nn.Module):
    """
    LoRA injection for nn.Linear layers.
    """
    def __init__(self, original_linear, rank, alpha=1.0, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.normal_(self.lora_B, std=0.02)

        self.scaling = alpha / rank

    def forward(self, x):
        original_out = self.original_linear(x)

        lora_out = x @ self.lora_B.t() 
        lora_out = lora_out @ self.lora_A.t() 
        lora_out = self.scaling * self.dropout(lora_out)

        return original_out + lora_out


class LoRAInjectionConv1D(nn.Module):
    def __init__(self, original_conv1d, rank, alpha=1.0, dropout=0.0):
        super().__init__()
        self.original_conv1d = original_conv1d
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


        in_features = self.original_conv1d.weight.shape[0]
        out_features = self.original_conv1d.weight.shape[1]

        self.lora_A = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.normal_(self.lora_B, std=0.02)

        self.scaling = alpha / rank

    def forward(self, x):
        size_out = x.size()[:-1] + (self.original_conv1d.weight.size(1),)
        x_2d = x.view(-1, x.size(-1))  

        original_out_2d = torch.addmm(
            self.original_conv1d.bias,  
            x_2d,                       
            self.original_conv1d.weight 
        )

        lora_out_2d = x_2d @ self.lora_B.t() 
        lora_out_2d = lora_out_2d @ self.lora_A.t() 
        lora_out_2d = self.scaling * self.dropout(lora_out_2d)

        final_2d = original_out_2d + lora_out_2d
        final = final_2d.view(*size_out) 
        return final


class AdapterInjection(nn.Module):
    def __init__(self, original_linear, adapter_size, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        self.adapter = nn.Sequential(
            nn.Linear(original_linear.out_features, adapter_size),
            nn.ReLU(),
            nn.Linear(adapter_size, original_linear.out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.original_linear(x)
        return out + self.adapter(out)


class PrefixInjection(nn.Module):
    def __init__(self, original_linear, hidden_size, scale=1.0, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        self.prefix = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.scale = scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.original_linear(x)
        prefix = self.dropout(self.prefix) * self.scale
        return out + prefix

def apply_lora_to_model(model, rank, alpha, dropout):
    for name, module in model.named_modules():
        # Handle nn.Linear
        if isinstance(module, nn.Linear):
            injected_module = LoRAInjection(
                module, rank=rank, alpha=alpha, dropout=dropout
            )

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model
            if parent_name:
                for level in parent_name.split("."):
                    parent_module = getattr(parent_module, level)
            setattr(parent_module, child_name, injected_module)

        elif isinstance(module, Conv1D):
            injected_module = LoRAInjectionConv1D(
                module, rank=rank, alpha=alpha, dropout=dropout
            )

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model
            if parent_name:
                for level in parent_name.split("."):
                    parent_module = getattr(parent_module, level)
            setattr(parent_module, child_name, injected_module)

    return model


def apply_adapters_to_model(model, adapter_size, dropout):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            injected_module = AdapterInjection(module, adapter_size=adapter_size, dropout=dropout)

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model
            if parent_name:
                for level in parent_name.split("."):
                    parent_module = getattr(parent_module, level)
            setattr(parent_module, child_name, injected_module)
    return model


def apply_prefix_layers_to_model(model, prefix_size, dropout):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # or any strategic place like wte, ln_1 etc.
            injected_module = PrefixInjection(
                module, hidden_size=module.out_features,
                scale=prefix_size, dropout=dropout
            )

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model
            if parent_name:
                for level in parent_name.split("."):
                    parent_module = getattr(parent_module, level)
            setattr(parent_module, child_name, injected_module)
            break  # optional: limit to just one place to inject prefix
    return model


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


    return parser.parse_args()


def main():

    args = parse_arguments()

    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        model = load_weight(model, state_dict)
        print("Model weights loaded successfully.")
    else:
        print('Weights gpt2-pytorch_model.bin not found')
        sys.exit()

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

    print("Model architecture successfully modified and ready for fine-tuning.")


if __name__ == "__main__":
    main()

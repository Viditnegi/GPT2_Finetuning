
# GPT-2 Fine-Tuning Toolkit

## Prerequisites

Create the conda environment:

```bash
conda create -n gpt2 python=3.8
conda activate gpt2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download pretrained GPT-2 weights:

```bash
curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
```

---

## Training with Different Fine-Tuning Techniques

### Adapter Tuning

```bash
python train_llm.py \
  --finetuning_technique adapters \
  --device cuda \
  --adapter_size <size> \
  --adapter_dropout <dropout_rate>
```

### LoRA Tuning

```bash
python train_llm.py \
  --finetuning_technique lora \
  --device cuda \
  --lora_rank <rank> \
  --lora_alpha <alpha> \
  --lora_dropout <dropout_rate>
```

### Prefix Tuning

```bash
python train_llm.py \
  --finetuning_technique prefix \
  --device cuda \
  --prefix_size <size> \
  --prefix_dropout <dropout_rate>
```

---

## Additional Parameters

You can add the following optional parameters to any of the above commands:

- `--num_epochs <int>`
- `--batch_size <int>`
- `--learning_rate <float>`
- `--num_dataloader_workers <int>`
- `--device <cpu|cuda>`

---

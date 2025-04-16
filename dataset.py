import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List
from gpt_2_Pytorch.GPT2.encoder import get_encoder


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

class AlpacaGPT2Dataset(Dataset):

    def __init__(self, split="train", max_length=256, from_csv=False):
    
        super().__init__()
        if(from_csv):
            self.data = pd.read_csv("alpaca.csv")
        else:
            self.data = load_dataset("tatsu-lab/alpaca", split=split)
        
        self.enc = get_encoder(
            json_path="gpt_2_Pytorch/GPT2/encoder.json",
            bpe_path="gpt_2_Pytorch/GPT2/vocab.bpe"
        )  
        self.max_length = max_length

        self.eos_token = "<|endoftext|>"
        self.eos_id = self.enc.encoder.get(self.eos_token, 50256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        record = self.data.iloc[idx]
        instruction = record["instruction"]
        inp = record["input"]
        out = record["output"]

        text = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=inp,
            response=out
        )

        input_ids = self.enc.encode(text)

        input_ids.append(self.eos_id)


        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]


        return {
            "input_ids": input_ids,
            "original_text": text 
        }

def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    eos_id = 50256 

    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    original_texts = []

    for item in batch:
        ids = item["input_ids"]        
        seq_len = len(ids)

        labels = ids[1:] + [eos_id]     

        pad_size = max_len - seq_len

        padded_input_ids.append(ids + [eos_id] * pad_size)
        padded_labels.append(labels + [-100]    * pad_size)
        attention_masks.append([1] * seq_len + [0] * pad_size)
        original_texts.append(item["original_text"])

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels,    dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks,  dtype=torch.long),
        "original_text": original_texts
    }


def main():
    dataset = AlpacaGPT2Dataset(
        split="train",
        max_length=256,
        from_csv=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    num_samples = 5
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n#################### BATCH {batch_idx} ####################")
        print("input_ids shape:       ", batch["input_ids"].shape)
        print("labels shape:          ", batch["labels"].shape)
        print("attention_mask shape:  ", batch["attention_mask"].shape)

        first_input_ids = batch["input_ids"][0].tolist()
        first_labels = batch["labels"][0].tolist()
        first_att_mask = batch["attention_mask"][0].tolist()
        first_original = batch["original_text"][0]

        print("\nFirst sample input_ids (first 30):", first_input_ids[:30])
        print("First sample labels (first 30):   ", first_labels[:30])
        print("First sample attention_mask(30):  ", first_att_mask[:30])

        print("\nOriginal prompt:\n", first_original)

        enc = dataset.enc
        decoded_text = enc.decode(first_input_ids)
        print("\nDecoded text (from input_ids):\n", decoded_text)

        if batch_idx == num_samples:
            break

if __name__ == "__main__":
    main()

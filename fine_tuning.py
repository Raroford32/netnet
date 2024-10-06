import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import create_directory_if_not_exists

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data.iloc[idx]['code']
        inputs = self.tokenizer.encode_plus(
            code,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def fine_tune_model(model, tokenizer, data, learning_rate=1e-5, batch_size=32, epochs=3, output_path="./finetuned_cpp_model", gradient_accumulation_steps=4):
    local_rank = setup_ddp()
    
    dataset = CodeDataset(data, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(tqdm(dataloader, desc="Training", disable=local_rank != 0)):
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    if local_rank == 0:
        # Save the fine-tuned model
        create_directory_if_not_exists(output_path)
        model.module.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Fine-tuned model saved to {output_path}")

    dist.destroy_process_group()
    return model

if __name__ == "__main__":
    from model_setup import setup_model
    from data_processor import download_and_preprocess_data

    model, tokenizer = setup_model()
    data = download_and_preprocess_data()
    fine_tune_model(model, tokenizer, data)

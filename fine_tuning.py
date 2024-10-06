import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import create_directory_if_not_exists

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
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

def fine_tune_model(model, tokenizer, data, learning_rate=5e-5, batch_size=4, epochs=3, output_path="./fine_tuned_model"):
    dataset = CodeDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Save the fine-tuned model
    create_directory_if_not_exists(output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Fine-tuned model saved to {output_path}")
    return model

if __name__ == "__main__":
    from model_setup import setup_model
    from data_processor import download_and_preprocess_data

    model, tokenizer = setup_model()
    data = download_and_preprocess_data()
    fine_tune_model(model, tokenizer, data)

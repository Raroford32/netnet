import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from fine_tuning import CodeDataset

def evaluate_model(model, tokenizer, data, batch_size=4):
    dataset = CodeDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"Evaluation results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    return avg_loss, perplexity

if __name__ == "__main__":
    from model_setup import setup_model
    from data_processor import download_and_preprocess_data

    model, tokenizer = setup_model(load_fine_tuned=True)
    data = download_and_preprocess_data()
    evaluate_model(model, tokenizer, data)

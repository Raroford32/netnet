from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_model(model_path="microsoft/CodeGPT-small-cpp", load_fine_tuned=False):
    if load_fine_tuned:
        model_path = "./fine_tuned_model"

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model loaded on device: {device}")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = setup_model()
    print("Model and tokenizer set up successfully.")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "microsoft/CodeGPT-small-cpp"
FINE_TUNED_MODEL_PATH = "./fine_tuned_model"

def setup_model(load_fine_tuned=False):
    if load_fine_tuned:
        model_path = FINE_TUNED_MODEL_PATH
    else:
        model_path = MODEL_NAME

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

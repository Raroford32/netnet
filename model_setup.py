import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_model(model_path="/home/ubuntu/mlabonne_Hermes-3-Llama-3.1-70B-lorablated", load_fine_tuned=False):
    if load_fine_tuned:
        model_path = "./finetuned_cpp_model"

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"Model loaded on GPU")
    else:
        print(f"Model loaded on CPU")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = setup_model()
    print("Model and tokenizer set up successfully.")

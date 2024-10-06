import torch

def generate_code(model, tokenizer, prompt, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code

if __name__ == "__main__":
    from model_setup import setup_model

    model, tokenizer = setup_model(load_fine_tuned=True)
    prompt = "// Write a C++ function to calculate the factorial of a number"
    generated_code = generate_code(model, tokenizer, prompt)
    print("Generated C++ code:")
    print(generated_code)

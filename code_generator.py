import torch

def generate_code(model, tokenizer, prompt, max_length=2048, temperature=0.8, top_p=0.95, repetition_penalty=1.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Enhance the prompt to encourage more complex and complicated C++ code
    enhanced_prompt = f"Generate a complex and sophisticated C++ implementation for the following task:\n{prompt}\n\nInclude advanced C++ features, optimizations, and best practices in your solution.\n\nC++ code:"

    input_ids = tokenizer.encode(enhanced_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the C++ code part
    code_start = generated_code.find("C++ code:")
    if code_start != -1:
        generated_code = generated_code[code_start + 9:].strip()

    return generated_code

if __name__ == "__main__":
    from model_setup import setup_model

    model, tokenizer = setup_model(load_fine_tuned=True)
    prompt = "Implement a concurrent hash map with lock-free read operations"
    generated_code = generate_code(model, tokenizer, prompt)
    print("Generated C++ code:")
    print(generated_code)

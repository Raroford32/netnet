import argparse
from data_processor import download_and_preprocess_data
from model_setup import setup_model
from fine_tuning import fine_tune_model
from evaluation import evaluate_model
from code_generator import generate_code

def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ").strip()
    return user_input if user_input else default_value

def main():
    parser = argparse.ArgumentParser(description="C++ Code Generation LLM Fine-Tuning Pipeline")
    parser.add_argument("--action", choices=["train", "generate"], required=True, help="Action to perform")
    parser.add_argument("--prompt", type=str, help="Prompt for code generation")
    args = parser.parse_args()

    if args.action == "train":
        print("Welcome to the C++ Code Generation LLM Fine-Tuning Pipeline!")
        print("Please provide the following inputs (press Enter to use default values):")

        model_path = get_user_input("Model path", "microsoft/CodeGPT-small-cpp")
        dataset_path = get_user_input("Dataset path", "./dataset")
        learning_rate = float(get_user_input("Learning rate", "5e-5"))
        batch_size = int(get_user_input("Batch size", "4"))
        num_epochs = int(get_user_input("Number of epochs", "3"))
        output_path = get_user_input("Output path for the fine-tuned model", "./fine_tuned_model")

        print("\nStarting the automated process...")

        print("Downloading and preprocessing data...")
        data = download_and_preprocess_data(dataset_path)

        print("Setting up the model...")
        model, tokenizer = setup_model(model_path)

        print("Fine-tuning the model...")
        fine_tuned_model = fine_tune_model(model, tokenizer, data, learning_rate, batch_size, num_epochs, output_path)

        print("Evaluating the model...")
        evaluate_model(fine_tuned_model, tokenizer, data)

        print("Training complete!")

        while True:
            generate_option = input("Do you want to generate C++ code using the fine-tuned model? (y/n): ").strip().lower()
            if generate_option == 'y':
                prompt = input("Enter a prompt for code generation: ")
                generated_code = generate_code(fine_tuned_model, tokenizer, prompt)
                print("Generated C++ code:")
                print(generated_code)
            elif generate_option == 'n':
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    elif args.action == "generate":
        if not args.prompt:
            args.prompt = input("Please provide a prompt for code generation: ")

        print("Loading the fine-tuned model...")
        model, tokenizer = setup_model(load_fine_tuned=True)

        print("Generating C++ code...")
        generated_code = generate_code(model, tokenizer, args.prompt)
        print("Generated C++ code:")
        print(generated_code)

if __name__ == "__main__":
    main()

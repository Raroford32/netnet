import argparse
from data_processor import download_and_preprocess_data
from model_setup import setup_model
from fine_tuning import fine_tune_model
from evaluation import evaluate_model
from code_generator import generate_code

def main():
    parser = argparse.ArgumentParser(description="C++ Code Generation LLM Fine-Tuning Pipeline")
    parser.add_argument("--action", choices=["train", "generate"], required=True, help="Action to perform")
    parser.add_argument("--prompt", type=str, help="Prompt for code generation")
    args = parser.parse_args()

    if args.action == "train":
        print("Downloading and preprocessing data...")
        data = download_and_preprocess_data()

        print("Setting up the model...")
        model, tokenizer = setup_model()

        print("Fine-tuning the model...")
        fine_tuned_model = fine_tune_model(model, tokenizer, data)

        print("Evaluating the model...")
        evaluate_model(fine_tuned_model, tokenizer, data)

        print("Training complete!")

    elif args.action == "generate":
        if not args.prompt:
            raise ValueError("Please provide a prompt for code generation.")

        print("Loading the fine-tuned model...")
        model, tokenizer = setup_model(load_fine_tuned=True)

        print("Generating C++ code...")
        generated_code = generate_code(model, tokenizer, args.prompt)
        print("Generated C++ code:")
        print(generated_code)

if __name__ == "__main__":
    main()

import argparse
import os
from data_processor import download_and_preprocess_data
from model_setup import setup_model
from fine_tuning import fine_tune_model
from evaluation import evaluate_model
from code_generator import generate_code

def main():
    parser = argparse.ArgumentParser(description="C++ Code Generation LLM Fine-Tuning Pipeline")
    parser.add_argument("--action", choices=["train", "generate"], default="train", help="Action to perform")
    parser.add_argument("--prompt", type=str, help="Prompt for code generation")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/mlabonne_Hermes-3-Llama-3.1-70B-lorablated", help="Path to the pre-trained model")
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to save/load the dataset")
    parser.add_argument("--output_path", type=str, default="./finetuned_cpp_model", help="Path to save the fine-tuned model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    args = parser.parse_args()

    if args.action == "train":
        print("Starting the automated C++ Code Generation LLM Fine-Tuning Pipeline...")

        print("Step 1: Downloading and preprocessing data...")
        data = download_and_preprocess_data(args.dataset_path)

        print("Step 2: Setting up the model...")
        model, tokenizer = setup_model(args.model_path)

        print("Step 3: Fine-tuning the model...")
        fine_tuned_model = fine_tune_model(model, tokenizer, data, args.learning_rate, args.batch_size, args.epochs, args.output_path)

        print("Step 4: Evaluating the model...")
        evaluate_model(fine_tuned_model, tokenizer, data)

        print("Training complete!")

        print("Generating a sample C++ code...")
        sample_prompt = "Write a C++ function to implement a binary search tree"
        generated_code = generate_code(fine_tuned_model, tokenizer, sample_prompt)
        print("Generated C++ code:")
        print(generated_code)

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

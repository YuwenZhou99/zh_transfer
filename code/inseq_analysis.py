import argparse
import inseq
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Run causal language model with specified parameters.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the causal language model to use.")
    parser.add_argument("--input_texts", type=str, nargs='+', required=True, help="Input texts for the model.")
    parser.add_argument("--generated_texts", type=str, nargs='+', required=True,
                        help="Generated texts for attribution.")

    args = parser.parse_args()

    # Load the specified causal language model and tokenizer
    model_name = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load the model with integrated gradients using inseq
    model = inseq.load_model(model_name, "integrated_gradients", tokenizer=tokenizer)

    # Perform attribution on the generated texts
    out = model.attribute(
        input_texts=args.input_texts,
        generated_texts=args.generated_texts,
        step_scores=["probability"],
    )

    # Display the attribution results
    out.show()


if __name__ == "__main__":
    main()

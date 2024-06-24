import argparse, glob, json
from transformers import AutoModelForMaskedLM, AutoTokenizer, MT5ForConditionalGeneration, T5Tokenizer
import os
from utils import run_masked_models, AvePplGoodBad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the masked language model to run")
    parser.add_argument("--metric", type=str, default="perplexity")
    args = parser.parse_args()

    # Read in SLING data
    sling_files = glob.glob("../data/SLING_Data/**/*.jsonl", recursive=True)

    for sling_file in sling_files:
        mp_dict_list = []
        dir = sling_file.split("/")
        phenomenon = dir[-2]
        paradigm = dir[-1].replace(".jsonl", "")
        good_sent, bad_sent = [], []

        with open(sling_file, "r", encoding='utf-8') as file:
            mp_dict_list.extend([json.loads(x) for x in file.read().strip().split("\n")])

        for mp_dict in mp_dict_list:
            good_sent.append(mp_dict["sentence_good"])
            bad_sent.append(mp_dict["sentence_bad"])

        print(f"LOADED\tPHENOMENON {phenomenon}\tPARADIGM {paradigm}")
        output_folder = f"outputs/SLING/CPM/{phenomenon}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = f"outputs/SLING/CPM/{phenomenon}/CPM_{paradigm}_sling.txt"
        with open(output_file, 'a+') as file:
            file.write(f"{phenomenon}\n\t{paradigm}\n")

        name = args.model_name
        print(f"*****Running {name}*****")

        if name == "google/mt5-small" or name == "google/mt5-large":
            model = MT5ForConditionalGeneration.from_pretrained(name)
            tokenizer = T5Tokenizer.from_pretrained(name)
        else:
            model = AutoModelForMaskedLM.from_pretrained(name, return_dict_in_generate=True,
                                                         output_scores=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model.eval()
        model.cuda()
        accuracy, good_pppl, bad_pppl = run_masked_models(model, tokenizer, good_sent, bad_sent, metric=args.metric)

        if args.metric == "perplexity":
            ave_ppl_good, ave_ppl_bad = AvePplGoodBad(good_pppl, bad_pppl)
        else:
            ave_ppl_good, ave_ppl_bad = 0.0, 0.0

        print(f"Model: {name}, Accuracy: {accuracy*100:.5f}, Average Perplexity (Good Sentences): {ave_ppl_good:.5f}, Average Perplexity (Bad Sentences): {ave_ppl_bad:.5f}\n")
        with open(output_file, 'a+') as file:
            file.write(f"Model: {name}, Accuracy: {accuracy*100:.5f}, Average Perplexity (Good Sentences): {ave_ppl_good:.5f}, Average Perplexity (Bad Sentences): {ave_ppl_bad:.5f}\n")
            k = 0
            for j, (good_ppl, bad_ppl) in enumerate(zip(good_pppl, bad_pppl)):
                file.write(f"{k}: good sentence {j}, PPL: {good_ppl:.5f}\n")
                file.write(f"{k+1}: bad sentence {j}, PPL: {bad_ppl:.5f}\n")
                k += 2

if __name__ == "__main__":
    main()

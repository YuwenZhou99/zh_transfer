import argparse
import os
import glob
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import run_causal_models, AvePplGoodBad

def load_climp_dataset(filename, skip_header=True, extract=lambda x: x[-2]):
    with open(f'../data/CLiMP_corpus/CLiMP_corpus/{filename}', 'r', encoding='utf-8') as f:
        dataset = f.readlines()

    if skip_header:
        dataset = dataset[1:]

    dataset = [x.split(',') for x in dataset]

    # confirm all elements of dataset have same length
    assert all([len(x) == len(dataset[0]) for x in dataset])
    assert len(dataset) % 2 == 0

    good_sent = [extract(dataset[i]) for i in range(0, len(dataset), 2)]
    bad_sent = [extract(dataset[i + 1]) for i in range(0, len(dataset), 2)]

    return good_sent, bad_sent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the causal language model to run")
    args = parser.parse_args()

    temp_names = ["anaphor_agreement_gender_1000.csv", "ba_construction_1000.csv",
                  "binding_gender_1000.csv", "classifier_1000.csv",
                  "classifier_adj_1000.csv", "classifier_clause_1000.csv",
                  "coverb_instrument_1000.csv", "coverb_with_1000.csv",
                  "filler_gap_dependency_1000.csv", "head_final_clause_1000.csv",
                  "passive_formal_1000.csv", "verb_complement_direction_1000.csv",
                  "verb_complement_duration_1000.csv", "verb_complement_frequency_1000.csv",
                  "verb_complement_res_adj_1000.csv", "verb_complement_res_verb_1000.csv"]

    for temp in temp_names:
        print("#############", temp, "#############")
        good_sent, bad_sent = load_climp_dataset(temp, skip_header=True, extract=lambda x: x[-2])
        print(f"{len(good_sent)} good sentences and {len(bad_sent)} bad sentences are loaded.")

        output_folder = 'outputs'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_path = f"outputs/CPM/CPM_{temp}_climp.txt"
        with open(file_path, 'a+') as file:
            file.write(f"{temp}\n")

            name = args.model_name
            print(f"*****Running {name}*****")

            model = AutoModelForCausalLM.from_pretrained(name, return_dict_in_generate=True, output_scores=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

            model.eval()
            model.cuda()
            accuracy, good_ppl, bad_ppl = run_causal_models(model, tokenizer, good_sent, bad_sent)

            ave_ppl_good, ave_ppl_bad = AvePplGoodBad(good_ppl, bad_ppl)

            print(f"Model: {name}, Accuracy: {accuracy:.5f}, Average Perplexity (Good Sentences): {ave_ppl_good:.5f}, Average Perplexity (Bad Sentences): {ave_ppl_bad:.5f}\n")
            file.write(f"Model: {name}, Accuracy: {accuracy:.5f}, Average Perplexity (Good Sentences): {ave_ppl_good:.5f}, Average Perplexity (Bad Sentences): {ave_ppl_bad:.5f}\n")

            k = 0
            for j, (good_ppl, bad_ppl) in enumerate(zip(good_ppl, bad_ppl)):
                file.write(f"{k}: good sentence {j}, PPL: {good_ppl:.5f}\n")
                file.write(f"{k+1}: bad sentence {j}, PPL: {bad_ppl:.5f}\n")
                k += 2

if __name__ == "__main__":
    main()

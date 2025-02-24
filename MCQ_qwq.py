import os
import sys
import config
os.environ['HF_TOKEN'] = config.HF_TOKEN
os.environ['HF_HOME'] = config.HF_HOME

import argparse
import copy
import json
import pandas as pd
import sys
import random
from tqdm import tqdm

from parse_string import LlamaParser
from agents import AgentAction, HuggingfaceChatbot
from utils import *
import random
import numpy as np
import torch

from datasets import Dataset
from datasets import load_dataset, load_from_disk
import config
HF_MCQ_path = config.HF_MCQ_path
dataset_dict = {
    "hard": load_from_disk(os.path.join(HF_MCQ_path, f'MCQ_dict_hard')),
    "easy": load_from_disk(os.path.join(HF_MCQ_path, f'MCQ_dict_easy')),
    "medium": load_from_disk(os.path.join(HF_MCQ_path, f'MCQ_dict_medium'))
}
def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)




def main(args):
    set_seeds(args)
    log(str(args),args.log_path)
    dataset = dataset_dict[args.strategy]
    if args.api_name:
        chatbot = ''
    else:    
        chatbot = HuggingfaceChatbot(args.model)
    agents = AgentAction(chatbot, 
                         parser_fn = LlamaParser().parse_MCQ,
                         template = args.prompt_template,
                         **vars(args))
    result_save_path = args.log_path.replace('.txt', '_results.txt')

    results = []
    # for domain in ['AI_ACT', 'GDPR', 'HIPAA']:
    # for domain in ['GDPR', 'HIPAA']:
    for domain in ['HIPAA', 'GDPR']:
        domain_dataset = [item for item in dataset if item.get("domain") == domain]
        if args.sample:
            domain_dataset = random.sample(domain_dataset, min(len(domain_dataset), args.sample))
        for i, item in enumerate(tqdm(domain_dataset)):
            label = item['label']
            decision = {}
            log(str(f"=== domain: {domain} --- case: {i}\n"), args.log_path)
            for _ in range(args.generation_round):
                try:
                    decision = agents.complete(**item)           
                    result = (decision["decision"] == label)
                    results.append(result)
                    acc = (sum(results) / len(results))
                    print(acc)
                    log(str(f"sample_id: {i} --- result:{result} --- answer: {decision['decision']}\n"), args.log_path)
                    log(str(decision)+"\n", args.log_path)
                    if decision: break

                except Exception as e:
                    print(e)
                    continue
            if not decision: results.append(0)

        acc = (sum(results) / len(results))
        print(acc)
        log(str(f"domain: {domain} --- num_sample: {len(dataset)} --- accuracy:{acc}\n"), args.log_path)
        log(str(f"domain: {domain} --- num_sample: {len(dataset)} --- accuracy:{acc}\n"), result_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--log_path", type=str, default="logs/MCQ_log.txt")
    parser.add_argument("--strategy", type=str, default='medium')
    parser.add_argument("--prompt_template", type=str, default="prompts/MCQ_template.txt")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--generation_round", type=int, default=10)
    parser.add_argument("--max_law_items", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_name", type=str, default='')
    ### newly appeneded
    # parser.add_argument("--domains", type=str, default='AI_ACT')
    parser.add_argument("--api_model", type=str, default=config.api_model)
    parser.add_argument("--api_token", type=str, default=config.api_key)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    main(args)

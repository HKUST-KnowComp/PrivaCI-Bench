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

from tqdm import tqdm

from parse_string import LlamaParser
from agents import AgentAction, HuggingfaceChatbot
from utils import *
import random
import numpy as np
import torch

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main(args):
    set_seeds(args)
    log(str(args),args.log_path)
    KBs = get_local_KB_dataset()
    cases = get_local_case_dataset()
    if args.api_name:
        chatbot = ''
    else:    
        chatbot = HuggingfaceChatbot(args.model)
    agents = AgentAction(chatbot, 
                         parser_fn = LlamaParser().parse_decision,
                         template = args.prompt_template,
                         **vars(args))
    result_save_path = args.log_path.replace('.txt', '_results.txt')
    for domain in args.domains.split('+'):
        assert domain in ['GDPR', 'HIPAA', 'AI_ACT', 'ACLU'], 'Invalid domain name' 
        KB_dataset = KBs[domain]
        case_dataset = cases[domain]
    #events = events[:5]
    ### if use api, replace chatbot with empty string
        predictions = []
        results = []
        for i, cur_case in enumerate(tqdm(case_dataset)):
            #if i > 2: break
            case_content = cur_case['case_content']
            norm_type = cur_case['norm_type']
            # ci elements
            args.sender = cur_case['sender']
            args.sender_role=cur_case['sender_role']
            args.recipient=cur_case['recipient']
            args.recipient_role=cur_case['recipient_role']
            args.subject=cur_case['subject']
            args.subject_role=cur_case['subject_role']
            args.information_type=cur_case['information_type']
            args.consent_form=cur_case['consent_form']
            args.purpose=cur_case['purpose']
            # if add clauses
            args.clauses=cur_case['followed_articles']+cur_case['violated_articles']

            label_list = label_transform(norm_type)
            decision = {}
            log(str(f"=== domain: {domain} --- case: {i}\n"), args.log_path)
            for _ in range(args.generation_round):
                try:
                    decision = agents.complete(event=case_content, domain = domain, **vars(args))           
                    result = decision["decision"].lower() in label_list
                    results.append(result)
                    # print(results[-1])
                    acc = (sum(results) / len(results))
                    print(acc)
                    log(str(f"sample_id: {i} --- result:{result} --- answer: {norm_type}\n"), args.log_path)
                    log(str(decision)+"\n", args.log_path)
                    if decision: break

                except Exception as e:
                    print(e)
                    continue
            if not decision: results.append(0)

        acc = (sum(results) / len(results))
        print(acc)
        log(str(f"domain: {domain} --- num_sample: {len(case_dataset)} --- accuracy:{acc}\n"), args.log_path)
        log(str(f"domain: {domain} --- num_sample: {len(case_dataset)} --- accuracy:{acc}\n"), result_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--log_path", type=str, default="logs/log.txt")
    parser.add_argument("--prompt_template", type=str, default="prompts/direct_answer_prompt.txt")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--generation_round", type=int, default=10)
    parser.add_argument("--max_law_items", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_name", type=str, default='')
    ### newly appeneded
    parser.add_argument("--domains", type=str, default='AI_ACT+GDPR+HIPAA+ACLU')
    parser.add_argument("--api_model", type=str, default=config.api_model)
    parser.add_argument("--api_token", type=str, default=config.api_key)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    main(args)

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)




import argparse
import copy
import json
import pandas as pd


from tqdm import tqdm

from parse_string import LlamaParser
from agents import AgentContentSearch, HuggingfaceChatbot
from utils import *

import random
import numpy as np
import torch




def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def KB_to_dict(kb):
    kb_dict = {}
    for item in kb:
        ### convert regulation id to lower case
        key = item['regulation_id']
        key = key.lower()
        kb_dict[key] = item
    return kb_dict


def main(args):
    set_seeds(args)
    log(str(args)+"\n",args.log_path)
    KBs = get_local_KB_dataset()
    cases = get_local_case_dataset()

    #events = events[:5]
    ### if use api, replace chatbot with empty string
    if args.api_name:
        chatbot = ''
    else:    
        chatbot = HuggingfaceChatbot(args.model)

    result_save_path = args.log_path.replace('.txt', '_results.txt')
    for domain in args.domains.split('+'):
        if domain == 'GDPR' or domain == 'HIPAA':
                continue
        assert domain in ['GDPR', 'HIPAA', 'AI_ACT'], 'Invalid domain name' 
        KB_dataset = KBs[domain]
        case_dataset = cases[domain]
        kb = KB_to_dict(KB_dataset)
        args.kb = kb
        args.domain = domain
        
        parser = LlamaParser(domain = args.domain)

        agents = AgentContentSearch(chatbot, args, parser)
        predictions = []
        results = []
        ### new appened for continuing eval from errors
        #ids,accs = parse_log(args.log_path)
        #last_id = ids[-1] if ids else -1
        last_id = -1
        if(last_id != -1):
            acc = accs[-1]
            correct = round(acc*(last_id+1))
            results = [0] * last_id
            results.append(correct)
            print(f'last_id: {last_id} with acc {acc}, total correct: {correct}')
        else:
            print('start from index 0')

        for i, cur_case in enumerate(tqdm(case_dataset)):
            #if i <= last_id:
            #    continue
            #if domain == 'AI_ACT' and i <= 2256:
            #    continue
            #if i > 3:
            #    break
            case_content = cur_case['case_content']
            norm_type = cur_case['norm_type']
            label_list = label_transform(norm_type)
            decision = {}
            log(str(f"=== domain: {domain} --- case: {i}\n"), args.log_path)
            #event = events.loc[i]
            decision = agents.action(case_content)
            decision["id"] = i
            if not "decision" in  decision:
                results.append(0)
                continue
            result = decision["decision"].lower() in label_list
            results.append(result)
            log(str(f"sample_id: {i} --- result:{result} --- answer: {norm_type}\n"), args.log_path)
            print(sum(results) / len(results))
            log(str(decision)+"\n", args.log_path)
        acc = (sum(results) / len(results))
        #log(str(f"accuracy:{acc}"), args.log_path)
        log(str(f"domain: {domain} --- num_sample: {len(case_dataset)} --- accuracy:{acc}\n"), args.log_path)
        log(str(f"domain: {domain} --- num_sample: {len(case_dataset)} --- accuracy:{acc}\n"), result_save_path)


def parse_log(log_path):
    import ast
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except:
        return []
    
    results = []
    acc = []
    for line in lines:
        if "{" == line[0]:
            cur_dict = ast.literal_eval(line.strip())
            id = cur_dict["id"]
            results.append(id)
        if line.startswith("0."):
            acc.append(float(line.strip()))
    return results, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--log_path", type=str, default=os.path.join('logs','log.txt'))


    parser.add_argument("--law_template", type=str, default="prompts/cot-knowledge-lookup-prompt.txt")
    parser.add_argument("--law_filter_template", type=str, default="prompts/3-beam-law-filter-prompt.txt")
    #parser.add_argument("--law_judge_template", type=str, default="prompts/3-cot-judge-regulation-prompt.txt")
    ###3-judge-regulation-prompt.txt
    parser.add_argument("--law_judge_template", type=str,    
                           default="prompts/3-judge-regulation-prompt.txt")
    parser.add_argument("--decision_making_template", type=str, default="prompts/4-cot-decision-making-merge.txt")


    parser.add_argument("--lawyer_tokens", type=int, default=1024)
    parser.add_argument("--law_filter_tokens", type=int, default=512)
    parser.add_argument("--decision_tokens", type=int, default=512)
    parser.add_argument("--law_judge_tokens", type=int, default=512)

    parser.add_argument("--law_generation_round", type=int, default=3)
    parser.add_argument("--law_filtering_round", type=int, default=3)
    parser.add_argument("--generation_round", type=int, default=5)
    parser.add_argument("--max_law_items", type=int, default=3)
    parser.add_argument("--look_up_items", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)

    #parser.add_argument("--use_content", type=str, default='yes')

    parser.add_argument("--api_name", type=str, default='')
    ### newly appeneded
    parser.add_argument("--domains", type=str, default='GDPR+HIPAA+AI_ACT')
    parser.add_argument("--api_model", type=str, default=config.api_model)
    parser.add_argument("--api_token", type=str, default=config.api_key)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)


    args = parser.parse_args()


    main(args)
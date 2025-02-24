# Official Code Repo for PrivaCI-Bench

## Installation
## Config File
Please configure your API key, HF token, and cache path and log path in `config.py`


## Data Reader
```python
from datasets import load_from_disk
data_law = load_from_disk('HF_cache/KBs')
data_cases = load_from_disk('HF_cache/cases')
```
Law Data features:
```
['reference', 
'norm_type', 
'sender', 
'sender_role', 
'recipient', 
'recipient_role', 
'subject', 
'subject_role', 
'information_type', 
'consent_form', 
'purpose', 
'sender_is_subject', 
'recipient_is_subject',  
'regulation_id',  
'regulation_content']
```
Case Data features:
```
['norm_type', 
'sender', 
'sender_role', 
'recipient', 
'recipient_role', 
'subject', 
'subject_role', 
'information_type', 
'consent_form', 
'purpose', 
'followed_articles', 
'violated_articles', 
'case_content']
```

## Reproduction
For main results:
```
script='direct_answer.py'
# script='direct_answer_qwq.py'
# script='cot_auto_answer.py'
# script='search_content_for_answer.py'

model='Qwen/Qwen2.5-7B-Instruct'
# model='mistralai/Mistral-7B-Instruct-v0.2'
# model='meta-llama/Llama-3.1-8B-Instruct'
# model='Qwen/QwQ-32B-Preview'

python $script --log_path logs/dp/qwen2.txt --model $model --domain 'GDPR+HIPAA+AI_ACT+ACLU'

python $script --log_path logs/dp/openai.txt --domain 'GDPR+HIPAA+AI_ACT+ACLU' --api_model gpt-4o-mini --api_name gpt-4o-mini

python $script --log_path logs/dp/deepseek.txt --domain 'GDPR+HIPAA+AI_ACT+ACLU' --api_model deepseek --api_name deepseek --api_token xxx
```

For ablations:
```
model='Qwen/Qwen2.5-7B-Instruct'
# model='mistralai/Mistral-7B-Instruct-v0.2'
# model='meta-llama/Llama-3.1-8B-Instruct'

## DP+CI
python direct_answer.py --log_path logs/abalation/results_ci.txt --model $model --domain 'GDPR+HIPAA+AI_ACT+ACLU' --prompt_template 'prompts/direct_answer_prompt_with_ci_element.txt'

## DP+CI+LAW
python direct_answer.py --log_path logs/abalation/results_ci_law.txt --model $model --domain 'GDPR+HIPAA+AI_ACT+ACLU' --prompt_template 'prompts/direct_answer_prompt_with_ci_element_with_law.txt'
```

For MCQ:
```
mode='easy'
# mode='medium'
# mode='hard'
python MCQ_qwq.py  --log_path logs/MCQ/result.txt --model Qwen/QwQ-32B-Preview --strategy $mode --sample 3000
```
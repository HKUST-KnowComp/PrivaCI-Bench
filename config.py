import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging_path = 'log.txt'
openai_api = ''
api_key = ''
CACHE_DIR = ''

HF_TOKEN = ""
HF_HOME = ""
api_model ='gpt-4o-mini'

### paths for HF format datasets
HF_cases_path = os.path.join(BASE_DIR, 'HF_cache', 'cases')
HF_KBs_path = os.path.join(BASE_DIR, 'HF_cache', 'KBs')
HF_MCQ_path = os.path.join(BASE_DIR, 'HF_cache', 'MCQ')

#other paras
MAX_REFERENCE_NUM = 10
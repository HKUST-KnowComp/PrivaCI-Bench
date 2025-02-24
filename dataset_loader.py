import datasets
from datasets import load_dataset, load_from_disk
import config


def load_local_HF_dataset(path):
    '''
    Load a dataset from a local path
    '''
    dataset = load_from_disk(path)
    return dataset



if __name__ == '__main__':
    KB_path = config.HF_KBs_path
    case_path = config.HF_cases_path
    kb_data = load_local_HF_dataset(KB_path)
    case_data = load_local_HF_dataset(case_path)
    print('loaded')
    
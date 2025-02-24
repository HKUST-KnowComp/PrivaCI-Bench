## Read in the MCQ dataset_dict
cd checklist2
```python
from datasets import Dataset
from datasets import load_dataset, load_from_disk
import config
import os
cache_dir = config.CACHE_DIR
dataset_dict = {
    "hard": load_from_disk(os.path.join(cache_dir, 'MCQ', f'MCQ_dict_hard')),
    "easy": load_from_disk(os.path.join(cache_dir, 'MCQ', f'MCQ_dict_easy')),
    "medium": load_from_disk(os.path.join(cache_dir, 'MCQ', f'MCQ_dict_medium'))
}
for key, dataset in dataset_dict.items():
    print(f"{key}:")
    print(dataset)
    print()

```
## Fullfill the prompt template
```python 
from MCQ.MCQ_template import *
templates['prompt_UC'].format(**dataset_dict['medium'][10])
```
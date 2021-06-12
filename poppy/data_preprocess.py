import pandas as pd
import re
import os

from tqdm import tqdm

## Cleaning train raw dataset

train = open('./data/raw/train.crash').readlines()

train_ids = []
train_texts = []
train_labels = []

for id, line in tqdm(enumerate(train)):
    line = line.strip()
    if line.startswith("train_"):
        train_ids.append(id)
    elif line == "0" or line == "1":
        train_labels.append(id)

for id, lb in tqdm(zip(train_ids, train_labels)):
    line_id = train[id].strip()
    label = train[lb].strip()
    text = ' '.join(train[id + 1: lb])
    text = re.sub('\s+', ' ', text).strip()[1: -1].strip()
    train_texts.append(text)

train_df = pd.DataFrame({
    'id': train_ids, 
    'text': train_texts, 
    'label': train_labels
})

if not os.path.exists('./data'):
    os.makedirs('./data')

train_df.to_csv('./data/train.csv', encoding='utf-8', index=False)


## Clean test raw dataset

test = open("./data/raw/test.crash").readlines()

test_ids = []
test_texts = []

for id, line in tqdm(enumerate(test)):
    line = line.strip()
    if line.startswith("test_"):
        test_ids.append(id)

for i, id in tqdm(enumerate(test_ids)):
    if i >= len(test_ids) - 1:
        end = len(test)
    else:
        end = test_ids[i + 1]

    line_id = test[id].strip()
    text = re.sub('\s+', ' ', ' '.join(test[id + 1: end])).strip()[1:-1].strip()
    test_texts.append(text)

test_df = pd.DataFrame({
    'id': test_ids, 
    'text': test_texts
})

submission = pd.read_csv('./data/raw/sample_submission.csv', encoding='utf-8')
result = pd.concat([test_df, submission], axis=1, sort=False)

result.to_csv('./data/test.csv', encoding='utf-8', index=False)
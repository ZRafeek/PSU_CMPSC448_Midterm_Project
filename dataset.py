# Pre-process the dataset
import csv
import bisect
import re

# Create the vocabulary for the Neural Network models
from collections import Counter
from typing import Any
# Create the Dataset object that torch can use
import torch
from torch.utils.data import Dataset
# Pad responses to be the same size
from torch.nn.utils.rnn import pad_sequence

llm_models = ['ChatGPT', 'Gemini', 'Grok', 'DeepSeek', 'Qwen', 'Claude', 'Llama']
llm_labels = {
    "ChatGPT": 0,
    'Gemini': 1, 
    'Grok': 2, 
    'DeepSeek': 3, 
    'Qwen': 4, 
    'Claude': 5, 
    'Llama':6
}

# TXD-22 A Large-Scale Benchmark Dataset
# https://data.mendeley.com/datasets/prcjcggtjf/1
dataset = 'data/raw/TXD-22 A Large-Scale Benchmark Dataset.csv'
dataset_encoding = 'utf-8-sig'
responses = 'data/processed/responses.csv'
tableHeader = ["prompt","response","llm","category","prompt_id"]

dataset_categories = [
    "Computer Science" ,"Engineering & energy", "Economics & finance", "Sociology",
    "Business & Management", "Pharmacy & Medicine", "English Language", "Law & Legal Studies", "Botany",
    "Criminology", "Disaster Science", "Geography & Environment", "Graphic Design", "History", "Information Science",
    "International Relations", "Mass Communication & Journalism", "Nuclear Engineering", "Philosophy", "Physics",
    "Political Science", "Psychology", "Robotics & AI", "Statistics", "Film & Photography", "Theatre", "Tourism & Hospitality", 
    "Gender Studies", "Agriculture", "Zoology"
    ]

def pre_process_dataset():
    dataset_responses_per_model = 3000

    responses_file = open(responses, 'w', newline='', encoding=dataset_encoding)
    csv_writer = csv.writer(responses_file, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(tableHeader)

    prompt_categories = []
    pID_start = []
    with open('data/raw/prompt_categories.csv', 'r', newline='') as prompt_categories_file:
        reader = csv.reader(prompt_categories_file, delimiter=',', quotechar='\"')
        for row in reader:
            pID_start.append(int(row[0]) - 1)
            prompt_categories.append(row[2].strip())
    prompt_categories_file.close()

    with open(dataset, encoding=dataset_encoding, newline='') as csvfile:
        raw = csvfile.read()

        # preprocessing
        def remove_internal_newlines(match):
            text = match.group(0)
            cleaned = text.replace('\n', '\\n').replace('\t', "\\t")
            return cleaned

        cleaned_data = re.sub(r'"[^"]*"', remove_internal_newlines, raw, flags=re.DOTALL)


        spamreader = csv.reader(cleaned_data.splitlines(), delimiter=',', quotechar='\"')
        rows = 0
        for row in spamreader:
            rows+=1
            model = row[2]
            if model in llm_models:
                promptID = (rows-1)%dataset_responses_per_model
                prompt = row[0]
                response = row[1]
                
                # prompt,response,llm,category
                category_index = bisect.bisect_right(pID_start, promptID) - 1
                category = prompt_categories[category_index]
                csv_writer.writerow([prompt,response,model,category,promptID])
                pass
            pass
        pass
    
    responses_file.close()

    print("Finished Pre-processing!")

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_frequency=2):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, count in counter.items():
        if count >= min_frequency:
            vocab[word]= len(vocab)

    return vocab

def text_to_ids(text, vocab):
    return [
        vocab.get(word, vocab["<UNK>"])
        for word in tokenize(text)
    ]

class LLMResponseDataset(Dataset):

    def __init__(self, data, vocab, input_type='output'):
        self.data = data
        self.vocab = vocab
        self.input_type = input_type

        self.labels = llm_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        if self.input_type == 'prompt':
            text = row["prompt"]
        elif self.input_type == 'output':
            text = row["response"]
        elif self.input_type == 'combined':
            text = "[PROMPT] " + row["prompt"] + \
            " [RESPONSE] " + row["response"]
        else:
            raise ValueError('Invalid input_type given')

        tokens = [
            self.vocab.get(word, self.vocab["<UNK>"])
            for word in text.lower().split()
        ]

        label = self.labels[row["llm"]]

        return torch.tensor(tokens), torch.tensor(label)

    pass

def collate_fn(batch):
    sequences, labels = zip(*batch)

    sequences = pad_sequence(list(sequences), batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return sequences, labels

if __name__ == '__main__':
    pre_process_dataset()
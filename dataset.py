# Pre-process the dataset
import csv
import bisect
import re
import pandas as pd

# Create the vocabulary for the Neural Network models
from collections import Counter
# Create the Dataset object that torch can use
import torch
from torch.utils.data import Dataset
# Pad responses to be the same size
from torch.nn.utils.rnn import pad_sequence

llm_models = ['ChatGPT', 'Gemini', 'Grok', 'DeepSeek', 'Qwen', 'Claude', 'Llama']

# TXD-22 A Large-Scale Benchmark Dataset
# https://data.mendeley.com/datasets/prcjcggtjf/1
dataset = 'data/raw/TXD-22 A Large-Scale Benchmark Dataset.csv'
dataset_encoding = 'utf-8-sig'
responses = 'data/processed/TXD_responses.csv'
tableHeader = ["prompt","response","llm","category","prompt_id"]
dataset_categories = [
    "Computer Science" ,"Engineering & energy", "Economics & finance", "Sociology",
    "Business & Management", "Pharmacy & Medicine", "English Language", "Law & Legal Studies", "Botany",
    "Criminology", "Disaster Science", "Geography & Environment", "Graphic Design", "History", "Information Science",
    "International Relations", "Mass Communication & Journalism", "Nuclear Engineering", "Philosophy", "Physics",
    "Political Science", "Psychology", "Robotics & AI", "Statistics", "Film & Photography", "Theatre", "Tourism & Hospitality", 
    "Gender Studies", "Agriculture", "Zoology"
    ]

# LMSYS Judging LLM-as-a-judge with MT-Bench and Chatbot Arena
# https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
responses_rw = 'data/processed/LMSYS_responses.csv'
tableHeader_rw = ["prompt", "response", "llm"]

def pre_process_TXD_dataset():
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
    
    responses_file.close()

def get_clean_llm_name(model_name):
    m = model_name.lower()
    if 'gpt' in m: return llm_models[0]
    if 'gemini' in m: return llm_models[1]
    if 'grok' in m: return llm_models[2]
    if 'deepseek' in m: return llm_models[3]
    if 'qwen' in m: return llm_models[4]
    if 'claude' in m: return llm_models[5]
    if 'llama' in m: return llm_models[6]
    return None

def extract_first_assistant_response(conversation_list):
    # Dig through the conversation array to find the first AI turn
    if conversation_list is None:
        return ""
    for msg in conversation_list:
        if msg.get('role') == 'assistant':
            return msg.get('content', '')
    return ""

lmsys_llms = {
    "ChatGPT":0,
    "Claude":0,
    "Llama":0
}
def pre_process_LMSYS_dataset():
    df = pd.read_parquet("hf://datasets/lmsys/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet")
    rows = []

    # preprocessing
    def remove_internal_newlines(match):
        text = match.group(0)
        cleaned = text.replace('\n', '\\n').replace('\t', "\\t")
        return cleaned

    for _, row in df.iterrows():
        if row['language'] != 'English': continue
        if 'prompt' in row and pd.notna(row['prompt']):
            prompt_text = row['prompt'].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
        else:
            prompt_text = row['conversation_a'][0]['content'].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "") if len(row['conversation_a']) > 0 else ''

        llm_a = get_clean_llm_name(row['model_a'])
        llm_b = get_clean_llm_name(row['model_b'])

        if llm_a and lmsys_llms[llm_a] < 1787:
            response_a = extract_first_assistant_response(row['conversation_a'])
            # cleaned_data = re.sub(r'"[^"]*"', remove_internal_newlines, response_a, flags=re.DOTALL)
            rows.append({"prompt":prompt_text, "response": response_a.replace("\n", "\\n").replace("\t", "\\t").replace("\r", ""), "llm":llm_a})
            lmsys_llms[llm_a] = lmsys_llms[llm_a]+1

        if llm_b and lmsys_llms[llm_b] < 1787:
            response_b = extract_first_assistant_response(row['conversation_b'])
            # cleaned_data = re.sub(r'"[^"]*"', remove_internal_newlines, response_b, flags=re.DOTALL)
            rows.append({"prompt":prompt_text, "response": response_b.replace("\n", "\\n").replace("\t", "\\t").replace("\r", ""), "llm":llm_b})
            lmsys_llms[llm_b] = lmsys_llms[llm_b]+1

    print(lmsys_llms)
    formatted_df = pd.DataFrame(rows, columns=tableHeader_rw)

    formatted_df.to_csv(responses_rw, index=False, encoding=dataset_encoding)


    pass

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

    def __init__(self, data, vocab, llm_labels, input_type='output'):
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

def collate_fn_cnn(batch):
    sequences, labels = zip(*batch)

    sequences = pad_sequence(list(sequences), batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return sequences, labels

def collate_fn_ltsm(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    sequences = pad_sequence(
        list(sequences),
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return sequences, lengths, labels

if __name__ == '__main__':
    pre_process_TXD_dataset()
    print("Finished Pre-processing the TXD Dataset!")
    pre_process_LMSYS_dataset()
    print("Finished Pre-processing the LMSYS Dataset!")
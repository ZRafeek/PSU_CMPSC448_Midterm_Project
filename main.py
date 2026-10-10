# import argparse
import random

import torch
import pandas as pd
import numpy as np

from models.modelArgs import ModelArguments, ModelType

from models.cnn import CNN
from models.ltsm import LTSM
from torch.utils.data import DataLoader, Subset, random_split
from dataset import LLMResponseDataset, collate_fn, build_vocab
from dataset import responses as data_csv_file
from dataset import tableHeader, dataset_encoding
from evaluate import evaluate

from train import train_model

def set_seed(seed = 45):
    random.seed = seed
    np.random.seed = seed
    torch.manual_seed = seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runModel(args : ModelArguments):
    set_seed(args.seed)

    print("Setting up LLM Data")
    llm_data = pd.read_csv(data_csv_file, encoding=dataset_encoding)
    llm_data = llm_data.dropna(subset=tableHeader)

    label_names = sorted(llm_data["llm"].unique())
    label_to_id = {
        label: index for index, label in enumerate(label_names)
    }

    llm_data["label"] = llm_data['llm'].map(label_to_id)

    if args.input == "output":
        texts = llm_data["response"].astype(str)
    elif args.input == "prompt":
        texts = llm_data["prompt"].astype(str)
    else:
        texts = (
            "Question: "+llm_data["prompt"].astype(str)+
            " Response: "+llm_data["response"].astype(str)
        )


    # Split Data Set
    print("Splitting Data Set by prompt_id")
    question_ids = llm_data["prompt_id"].unique().tolist()
    random.shuffle(question_ids)

    train_end = int(0.70 * len(question_ids))
    val_end = int(0.85 * len(question_ids))

    train_ids = set(question_ids[:train_end])
    val_ids = set(question_ids[train_end:val_end])
    test_ids = set(question_ids[val_end:])

    print("Getting data indicies")
    train_indicies = [
        i for i, qid in enumerate(llm_data["prompt_id"])
        if qid in train_ids
    ]
    val_indicies = [
        i for i, qid in enumerate(llm_data["prompt_id"])
        if qid in val_ids
    ]
    test_indicies = [
        i for i, qid in enumerate(llm_data["prompt_id"])
        if qid in test_ids
    ]

    # Build Vocabulary from training data
    print("Building vocabulary")
    training_data_list = texts.iloc[np.array(train_indicies)].tolist()
    vocab = build_vocab(training_data_list)
    
    dataset = LLMResponseDataset(
        data=llm_data,
        vocab=vocab,
        input_type=args.input
    )

    train_data_loader = DataLoader(Subset(dataset, train_indicies), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(Subset(dataset, val_indicies), batch_size=args.batch_size, collate_fn=collate_fn)
    test_data_loader = DataLoader(Subset(dataset, test_indicies), batch_size=args.batch_size, collate_fn=collate_fn)

    num_classes = len(label_names)
    vocab_size = len(vocab)

    if args.model == 'cnn':
        model = CNN(
            vocab_size=vocab_size,
            num_classes=num_classes
        )
    else:
        model = LTSM(
            vocab_size=vocab_size,
            num_classes=num_classes
        )

    if model == None:
        print("Model not initialized!!!")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Train Model
    print("Training Model...")
    train_model(model=model, train_loader=train_data_loader, val_loader=val_data_loader, device=device, epochs=args.epochs)

    # Evaluate
    print("\nResults")
    print(f"Architecture: {args.model}")
    print(f"Input Type: {args.input}")
    print(f"Classes: {label_names}")
    evaluate(model=model, loader=test_data_loader, device=device)


def main():
    RQ = 1
    args = ModelArguments()

    if RQ == 2:
        pass
    elif RQ == 3:
        pass
    else:
        print('Invalid RQ given, defaulting to RQ1')
    
    runModel(args)

if __name__ == "__main__":
    main()
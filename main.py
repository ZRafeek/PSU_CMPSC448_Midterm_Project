# import argparse
import random

import torch
import pandas as pd
import numpy as np

from models.modelArgs import ModelArguments, ModelType, InputType, TraningSeparatorType

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

def split_data(separator : TraningSeparatorType, llm_data, resultsFile):
    if separator == TraningSeparatorType.PROMPT:
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
    else:
        print("Splitting Data Set by category")
        category_label = llm_data["category"].unique().tolist()
        random.shuffle(category_label)

        train_end = int(0.70 * len(category_label))
        val_end = int(0.85 * len(category_label))

        train_ids = set(category_label[:train_end])
        val_ids = set(category_label[train_end:val_end])
        test_ids = set(category_label[val_end:])

        print("Getting data indicies")
        train_indicies = [
            i for i, qid in enumerate(llm_data["category"])
            if qid in train_ids
        ]
        val_indicies = [
            i for i, qid in enumerate(llm_data["category"])
            if qid in val_ids
        ]
        test_indicies = [
            i for i, qid in enumerate(llm_data["category"])
            if qid in test_ids
        ]
        resultsFile.write("Training Categories:\n")
        resultsFile.writelines([', '.join(train_ids), "\n"])
        resultsFile.write("Validation Categories:\n")
        resultsFile.writelines([', '.join(val_ids), "\n"])
        resultsFile.write("Test Categories:\n")
        resultsFile.writelines([', '.join(test_ids), "\n"])
    return train_indicies, val_indicies, test_indicies

def runModel(args : ModelArguments):
    resultsFile = open("results/"+args.fileName+".txt", 'w')
    
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
            "[PROMPT] "+llm_data["prompt"].astype(str)+
            " [RESPONSE] "+llm_data["response"].astype(str)
        )

    train_indicies, val_indicies, test_indicies = split_data(args.trainingSeparator, llm_data, resultsFile)

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

    if args.model == ModelType.CNN:
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
    resultsFile.write("Results:\n")
    resultsFile.writelines(["Architecture: ", args.model.upper(), "\n"])
    print(f"Input Type: {args.input}")
    resultsFile.writelines(["Input Type: ", args.input, "\n"])
    print(f"Classes: {label_names}")
    resultsFile.write("\nClasses:\n")
    resultsFile.writelines([", ".join(label_names)])
    resultsFile.write("\n")

    evaluation = evaluate(model=model, loader=test_data_loader, device=device, filename=args.fileName)
    print(evaluation)
    resultsFile.writelines([evaluation])

    resultsFile.close()
    torch.save(model.state_dict(), "results/"+args.fileName+".pt")


def main():
    RQ = 1
    args = ModelArguments()
    args.setEpochs(1)
    for RQ in range(1,4):
        if RQ == 1:
            '''
            RQ1 — Can we identify which LLM generated a response?
            Use LLM_output as input, conduct classification to detect which LLM
            generated it. Report the performance of your CNN & RNN performance
            '''
            args.setModel(ModelType.CNN)
            args.setFileName("RQ1_CNN")
            runModel(args)
            args.setModel(ModelType.LTSM)
            args.setFileName("RQ1_RNN")
            runModel(args)
        elif RQ == 2:
            '''
            RQ2 — Does the user's prompt help identify the LLM?
            Compare compare classifiers using: Input only; Output only; Input + Output
            Questions you may consider: 
            i) Can the LLM be predicted from the user's prompt even before seeing its response? 
            ii) Does including the input improve classification? 
            iii) If input-only classification works surprisingly well, why might this happen?
            '''
            args.setModel(ModelType.CNN)
            args.setFileName("RQ2_CNN_Prompt")
            args.setInput(InputType.PROMPT)
            runModel(args)

            args.setFileName("RQ2_CNN_Output")
            args.setInput(InputType.OUTPUT)
            runModel(args)

            args.setFileName("RQ2_CNN_Combined")
            args.setInput(InputType.COMBINED)
            runModel(args)

            args.setModel(ModelType.LTSM)
            args.setFileName("RQ2_RNN_Prompt")
            args.setInput(InputType.PROMPT)
            runModel(args)

            args.setFileName("RQ2_RNN_Output")
            args.setInput(InputType.OUTPUT)
            runModel(args)

            args.setFileName("RQ2_RNN_Combined")
            args.setInput(InputType.COMBINED)
            runModel(args)
        elif RQ == 3:
            '''
            RQ3 — Do LLM fingerprints generalize across tasks or domains?
            Investigate whether a classifier trained on one type of task can identify LLMs on another type of task.
            For example: Training on problems like coding + mathematics + factual questions, but Testing on writing. 
            Or Training on general question answering but Testing on coding.
            Questions you may consider: 
            i) Does performance decrease on unseen tasks? 
            ii) Are some models more consistently identifiable across domains? 
            iii) Is the classifier learning a general LLM fingerprint or task-specific patterns?
            '''
            args.setModel(ModelType.CNN)
            args.setFileName("RQ3_CNN")
            args.setInput(InputType.OUTPUT)
            args.setTrainingSeparator(TraningSeparatorType.CATEGORY)
            runModel(args)

            args.setModel(ModelType.CNN)
            args.setFileName("RQ3_RNN")
            args.setInput(InputType.OUTPUT)
            args.setTrainingSeparator(TraningSeparatorType.CATEGORY)
            runModel(args)
        else:
            print('Invalid RQ given, defaulting to Running base model')
    
    print("FINISHED")

if __name__ == "__main__":
    main()
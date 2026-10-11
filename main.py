import random
import time

import torch
import pandas as pd
import numpy as np

from models.modelArgs import ModelArguments, ModelType, InputType, TraningSeparatorType, DatasetType

from models.cnn import CNN
from models.ltsm import LTSM
from torch.utils.data import DataLoader, Subset
from dataset import LLMResponseDataset, collate_fn_cnn, collate_fn_ltsm, build_vocab

from dataset import responses as txd_data_csv_file
from dataset import tableHeader, tableHeader_rw, dataset_encoding

from dataset import responses_rw as lmsys_data_csv_file

from evaluate import evaluate
from sklearn.model_selection import train_test_split

from train import train_model

def set_seed(seed = 45):
    random.seed = seed
    np.random.seed = seed
    torch.manual_seed = seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_data(args : ModelArguments, llm_data):
    separator = args.trainingSeparator
    if separator == TraningSeparatorType.PROMPT or separator == TraningSeparatorType.CATEGORY:
        split = 'category' if separator == TraningSeparatorType.CATEGORY else 'prompt_id'
        print("Splitting Data Set by ",split)

        question_ids = llm_data[split].unique().tolist()
        random.shuffle(question_ids)

        train_end = int(0.70 * len(question_ids))
        val_end = int(0.85 * len(question_ids))

        train_ids = set(question_ids[:train_end])
        val_ids = set(question_ids[train_end:val_end])
        test_ids = set(question_ids[val_end:])

        print("Getting data indicies")
        train_indicies = [
            i for i, qid in enumerate(llm_data[split])
            if qid in train_ids
        ]
        val_indicies = [
            i for i, qid in enumerate(llm_data[split])
            if qid in val_ids
        ]
        test_indicies = [
            i for i, qid in enumerate(llm_data[split])
            if qid in test_ids
        ]

    else:
        print("Using Stratisfied data splits")
        data = llm_data.sample(frac = 1, random_state = args.seed).reset_index(drop=True)
        # random.shuffle(data)

        # Split 70% training, 30% temporary
        all_indices = list(range(len(data)))

        train_indicies, temp_indices = train_test_split(
            all_indices,
            test_size=0.30,
            random_state=args.seed,
            stratify=data["llm"]
        )

        # Split the remaining 30% into 15% validation and 15% test
        val_indicies, test_indicies = train_test_split(
            temp_indices,
            test_size=0.50,
            random_state=args.seed,
            stratify=data.iloc[temp_indices]["llm"]
        )

    return train_indicies, val_indicies, test_indicies

def load_dataset(dataset_type : DatasetType):
    if dataset_type == DatasetType.LMSYS:
        path = lmsys_data_csv_file
        column_names = tableHeader_rw
    else:
        path = txd_data_csv_file
        column_names = tableHeader

    df = pd.read_csv(path, encoding=dataset_encoding)
    missing = set(column_names) - set(df.columns)
    if missing:
        raise ValueError("Missing columns")

    df = df.dropna(subset=column_names).copy()

    for column in tableHeader_rw:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df['prompt'] != "") & (df['response'] != "") & (df['llm'] != "")
    ].reset_index(drop=True)

    return df

def prepare_shared_labels(source_df, target_df):
    source_labels = set(source_df['llm'].unique())
    target_labels = set(target_df['llm'].unique())

    shared_labels = sorted(source_labels & target_labels)

    if len(shared_labels) < 2:
        raise ValueError ("Need at least two shared classes for classification")

    print("Shared LLM classes", shared_labels)

    source_df = source_df[
        source_df["llm"].isin(shared_labels)
    ].reset_index(drop=True)

    target_df = target_df[
        target_df["llm"].isin(shared_labels)
    ].reset_index(drop=True)

    label_to_id = {
        label:index for index, label in enumerate(shared_labels)
    }

    return source_df, target_df, label_to_id

def runModel(args : ModelArguments):
    start_time = time.time()
    print("\nNow Running "+args.fileName+":")
    resultsFile = open("results/"+args.fileName+".txt", 'w')
    
    set_seed(args.seed)

    source_df = load_dataset(args.dataset)
    target_df = load_dataset(args.test_dataset)
    cross_dataset = args.dataset != args.test_dataset

    print("Setting up LLM Data")
    if cross_dataset:
        source_df, target_df, label_to_id = prepare_shared_labels(source_df, target_df)
    else:
        label_names = sorted(source_df["llm"].unique())
        label_to_id = {
            label: index for index, label in enumerate(label_names)
        }
    
        source_df["label"] = source_df['llm'].map(label_to_id)
        target_df["label"] = target_df['llm'].map(label_to_id)

    label_names = sorted(source_df["llm"].unique())

    if args.input == "output":
        source_texts = source_df["response"].astype(str)
    elif args.input == "prompt":
        source_texts = source_df["prompt"].astype(str)
    else:
        source_texts = (
            "[PROMPT] "+source_df["prompt"].astype(str)+
            " [RESPONSE] "+source_df["response"].astype(str)
        )

    train_indicies, val_indicies, test_indicies = split_data(args, source_df)

    # Build Vocabulary from training data
    print("Building vocabulary")
    training_data_list = source_texts.iloc[np.array(train_indicies)].tolist()
    vocab = build_vocab(training_data_list)

    source_dataset = LLMResponseDataset(
        data=source_df,
        vocab=vocab,
        input_type=args.input,
        llm_labels=label_to_id
    )

    target_dataset = LLMResponseDataset(
        data=target_df,
        vocab=vocab,
        input_type=args.input,
        llm_labels=label_to_id
    )

    collate = (collate_fn_cnn if args.model == ModelType.CNN else collate_fn_ltsm)

    train_data_loader = DataLoader(Subset(source_dataset, train_indicies), batch_size=args.batch_size, shuffle=True, 
                                collate_fn=collate)
    val_data_loader = DataLoader(Subset(source_dataset, val_indicies), batch_size=args.batch_size, 
                                collate_fn=collate)
    if cross_dataset:
        test_data_loader = DataLoader(target_dataset, batch_size=args.batch_size, 
                                    collate_fn=collate)
    else:
        test_data_loader = DataLoader(Subset(source_dataset, test_indicies), batch_size=args.batch_size, 
                                    collate_fn=collate)

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
    train_model(model=model, train_loader=train_data_loader, val_loader=val_data_loader, device=device, epochs=args.epochs, modelType=args.model)

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

    evaluation = evaluate(model=model, loader=test_data_loader, device=device, filename=args.fileName, labels=label_names, modelType=args.model)
    print(evaluation)
    resultsFile.writelines([evaluation])

    resultsFile.close()
    torch.save(model.state_dict(), "results/"+args.fileName+".pt")
    print(f"Model {args.fileName} took {time.time() - start_time} seconds")


def main():
    start_time = time.time()
    args = ModelArguments()
    # args.setEpochs(1)
    for RQ in range(1,4):
        if RQ == 1:
            '''
            RQ1 — Can we identify which LLM generated a response?
            Use LLM_output as input, conduct classification to detect which LLM
            generated it. Report the performance of your CNN & RNN performance
            '''
            args.setTrainingSeparator(TraningSeparatorType.PROMPT)
            args.setDataset(DatasetType.TXD)
            args.setTestDataset(DatasetType.TXD)

            args.setModel(ModelType.CNN)
            args.setFileName("RQ1/CNN")
            runModel(args)
            args.setModel(ModelType.LTSM)
            args.setFileName("RQ1/RNN")
            runModel(args)
        elif RQ == 2:
            '''
            RQ2 — Does the user's prompt help identify the LLM?
            Compare compare classifiers using: Input only; Output only; Input + Output
            Questions you may consider: 
            i) Can the LLM be predicted from the user's prompt even before seeing its response? 
            ii) Does including the input improve classification? 
            iii) If input-only classification works surprisingly well, why might this happen?

            ** Note that for more realistic values, a different dataset is used here.
            This data set only has responses from:
            ChatGPT: 8820
            Claude: 5574
            Llama: 1787

            For fairness, each model is limited to 1787 sets of data
            '''
            for dataset in DatasetType:
                args.setTrainingSeparator(TraningSeparatorType.PROMPT if dataset == DatasetType.TXD else TraningSeparatorType.STRATISFIED)
                args.setDataset(dataset)
                args.setTestDataset(dataset)

                args.setModel(ModelType.CNN)
                args.setFileName("RQ2/CNN_Prompt_"+dataset.name)
                args.setInput(InputType.PROMPT)
                runModel(args)

                args.setFileName("RQ2/CNN_Output_"+dataset.name)
                args.setInput(InputType.OUTPUT)
                runModel(args)

                args.setFileName("RQ2/CNN_Combined_"+dataset.name)
                args.setInput(InputType.COMBINED)
                runModel(args)

                args.setModel(ModelType.LTSM)
                args.setFileName("RQ2/RNN_Prompt_"+dataset.name)
                args.setInput(InputType.PROMPT)
                runModel(args)

                args.setFileName("RQ2/RNN_Output_"+dataset.name)
                args.setInput(InputType.OUTPUT)
                runModel(args)

                args.setFileName("RQ2/RNN_Combined_"+dataset.name)
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
            args.setTrainingSeparator(TraningSeparatorType.STRATISFIED)
            args.setInput(InputType.OUTPUT)

            args.setDataset(DatasetType.TXD)
            args.setTestDataset(DatasetType.LMSYS)

            args.setModel(ModelType.CNN)
            args.setFileName("RQ3/CNN_TrainTXD")
            runModel(args)

            args.setModel(ModelType.LTSM)
            args.setFileName("RQ3/RNN_TrainTXD")
            runModel(args)

            args.setDataset(DatasetType.LMSYS)
            args.setTestDataset(DatasetType.TXD)

            args.setModel(ModelType.CNN)
            args.setFileName("RQ3/CNN_TrainLMSYS")
            runModel(args)

            args.setModel(ModelType.LTSM)
            args.setFileName("RQ3/RNN_TrainLMSYS")
            runModel(args)
        else:
            print('Invalid RQ given, defaulting to Running base model')
        
    print("\nFINISHED")
    print(f"Took: {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
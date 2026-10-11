import random
import time

import torch
import pandas as pd
import numpy as np

from models.modelArgs import ModelArguments, ModelType, InputType, TraningSeparatorType

from models.cnn import CNN
from models.ltsm import LTSM
from torch.utils.data import DataLoader, Subset, random_split
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

def split_data(args : ModelArguments, llm_data, resultsFile):
    separator = args.trainingSeparator
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
    elif separator == TraningSeparatorType.CATEGORY:
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
    else:
        print("Using real-world data, no determinable splits possible")
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

def runModel(args : ModelArguments):
    start_time = time.time()
    print("\nNow Running "+args.fileName+":")
    resultsFile = open("results/"+args.fileName+".txt", 'w')
    
    set_seed(args.seed)

    print("Setting up LLM Data")
    llm_data = pd.read_csv(lmsys_data_csv_file if args.trainingSeparator == TraningSeparatorType.REAL_WORLD else txd_data_csv_file, encoding=dataset_encoding)
    
    llm_data = llm_data.dropna(subset=tableHeader_rw if args.trainingSeparator == TraningSeparatorType.REAL_WORLD else tableHeader)

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

    train_indicies, val_indicies, test_indicies = split_data(args, llm_data, resultsFile)

    # Build Vocabulary from training data
    print("Building vocabulary")
    training_data_list = texts.iloc[np.array(train_indicies)].tolist()
    vocab = build_vocab(training_data_list)
    
    dataset = LLMResponseDataset(
        data=llm_data,
        vocab=vocab,
        input_type=args.input,
        llm_labels=label_to_id
    )

    train_data_loader = DataLoader(Subset(dataset, train_indicies), batch_size=args.batch_size, shuffle=True, 
                                   collate_fn=collate_fn_cnn if args.model == ModelType.CNN else collate_fn_ltsm)
    val_data_loader = DataLoader(Subset(dataset, val_indicies), batch_size=args.batch_size, 
                                 collate_fn=collate_fn_cnn if args.model == ModelType.CNN else collate_fn_ltsm)
    test_data_loader = DataLoader(Subset(dataset, test_indicies), batch_size=args.batch_size, 
                                  collate_fn=collate_fn_cnn if args.model == ModelType.CNN else collate_fn_ltsm)

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
    # RQ = 1
    start_time = time.time()
    args = ModelArguments()
    # args.setEpochs(1)
    for RQ in range(1,4):
    # RQ = 2
        if RQ == 1:
            '''
            RQ1 — Can we identify which LLM generated a response?
            Use LLM_output as input, conduct classification to detect which LLM
            generated it. Report the performance of your CNN & RNN performance
            '''
            args.setTrainingSeparator(TraningSeparatorType.PROMPT)
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

            ** Note that for more realistic values, a different dataset is used here.
            This data set only has responses from:
            ChatGPT: 8820
            Claude: 5574
            Llama: 1787

            For fairness, each model is limited to 1787 sets of data
            '''
            args.setTrainingSeparator(TraningSeparatorType.REAL_WORLD)
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
            args.setTrainingSeparator(TraningSeparatorType.CATEGORY)
            args.setModel(ModelType.CNN)
            args.setFileName("RQ3_CNN")
            args.setInput(InputType.OUTPUT)
            runModel(args)

            args.setModel(ModelType.LTSM)
            args.setFileName("RQ3_RNN")
            args.setInput(InputType.OUTPUT)
            runModel(args)
        else:
            print('Invalid RQ given, defaulting to Running base model')
        
    print("\nFINISHED")
    print(f"Took: {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
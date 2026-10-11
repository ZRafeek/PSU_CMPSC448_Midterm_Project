from enum import StrEnum

class ModelType(StrEnum):
    CNN = 'cnn'
    LTSM = 'ltsm'

class InputType(StrEnum):
    OUTPUT = 'output'
    PROMPT = 'prompt'
    COMBINED = 'combined'

class TraningSeparatorType(StrEnum):
    PROMPT = 'prompt_id'
    CATEGORY = 'category'
    STRATISFIED = 'stratisfied'

class DatasetType(StrEnum):
    TXD = 'txd-22'
    LMSYS = 'lmsys'

class ModelArguments:

    def __init__(self):
        self.model = ModelType.CNN
        self.input = InputType.OUTPUT
        self.epochs = 10
        self.batch_size = 32
        self.seed = 45
        self.fileName = "Default"
        self.trainingSeparator = TraningSeparatorType.PROMPT
        self.dataset = DatasetType.TXD
        self.test_dataset = DatasetType.TXD

    def setModel(self, model:ModelType):
        self.model = model

    def setInput(self, input:InputType):
        self.input = input

    def setEpochs(self, epochs:int):
        self.epochs = epochs

    def setBatchSize(self, batch_size:int):
        self.batch_size = batch_size

    def setSeed(self, seed:int):
        self.seed = seed

    def setFileName(self, filename:str):
        self.fileName = filename
    
    def setTrainingSeparator(self, trainingSeparator:TraningSeparatorType):
        self.trainingSeparator = trainingSeparator

    def setDataset(self, dataset:DatasetType):
        self.dataset = dataset

    def setTestDataset(self, dataset:DatasetType):
        self.test_dataset = dataset
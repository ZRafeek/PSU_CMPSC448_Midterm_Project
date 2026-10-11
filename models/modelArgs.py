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
    REAL_WORLD = 'real_world'

class ModelArguments:

    def __init__(self):
        self.model = ModelType.CNN
        self.input = InputType.OUTPUT
        self.epochs = 10
        self.batch_size = 32
        self.seed = 45
        self.fileName = "Default"
        self.trainingSeparator = TraningSeparatorType.PROMPT

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
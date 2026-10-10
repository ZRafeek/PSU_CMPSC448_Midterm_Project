from enum import StrEnum

class ModelType(StrEnum):
    CNN = 'cnn'
    LTSM = 'ltsm'

class InputType(StrEnum):
    OUTPUT = 'output'
    PRMOPT = 'prompt'
    COMBINED = 'combined'

class ModelArguments:

    def __init__(self):
        self.model = 'cnn'
        self.input = 'output'
        self.epochs = 10
        self.batch_size = 32
        self.seed = 45

    def setModel(self, model:ModelType):
        models = ['cnn', 'ltsm']
        if model in models:
            self.model = model

    def setInput(self, input:InputType):
        inputs = ['output', 'prompt', 'combined']
        if input in inputs:
            self.input = input

    def setEpochs(self, epochs:int):
        self.epochs = epochs

    def setBatchSize(self, batch_size:int):
        self.batch_size = batch_size

    def setSeed(self, seed:int):
        self.seed = seed
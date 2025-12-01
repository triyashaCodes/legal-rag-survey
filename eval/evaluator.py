# Evaluator module

from tqdm import tqdm

class Evaluator:
    def __init__(self, framework_runner):
        self.runner = framework_runner

    def evaluate(self, dataset):
        predictions, gold = [], []

        for ex in tqdm(dataset):
            output = self.runner.ask(ex["question"])
            predictions.append(output)
            gold.append(ex["answer"])

        return predictions, gold

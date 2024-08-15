from .utils import *
from .model.model_process import *

class Tester():
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.test_processor = DevBatchProcessor(args, kg)

    def test(self):
        self.args.valid = False
        return self.test_processor.process_epoch(self.model)

class UnlearningTester():
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.forget_test_processor = ForgetDevBatchProcessor(args, kg)
        self.reserve_test_processor = ReserveDevBatchProcessor(args, kg)

    def test(self):
        self.args.valid = False
        forget_results = self.forget_test_processor.process_epoch(self.model)
        reserve_results = self.reserve_test_processor.process_epoch(self.model)
        return forget_results, reserve_results
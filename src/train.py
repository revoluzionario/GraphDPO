from .utils import *
from .model.model_process import *

class Trainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        self.train_processor = TrainBatchProcessor(args, kg)
        self.valid_processor = DevBatchProcessor(args, kg)

    def run_epoch(self):
        self.args.valid = True
        loss = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res


class UnlearningTrainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        if self.args.unlearning_method in ["repretrain", "Before", "Finetune"]:
            self.train_processor = TrainBatchProcessor(args, kg)
        elif self.args.unlearning_method == "NG":
            self.train_processor = NGTrainBatchProcessor(args, kg)
        elif self.args.unlearning_method == "RL":
            self.train_processor = RLTrainBatchProcessor(args, kg)
        elif self.args.unlearning_method == "Fisher":
            self.train_processor = FisherTrainBatchProcessor(args, kg)
        elif self.args.unlearning_method == "BS":
            self.train_processor = BSTrainBatchProcessor(args, kg)
        elif self.args.unlearning_method == "ADVIMP":
            self.train_processor = ADVIMPTrainBatchProcessor(args, kg, model)
            self.origin_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
            self.importance = self.estimate_parameter_importance(self.train_processor.data_loader, deepcopy(self.model), self.args.device, self.kg.snapshots[self.args.snapshot].forgotten_num, self.optimizer)
            self.train_processor.importance = self.importance
        elif self.args.unlearning_method == "SSD":
            self.train_processor = SSDTrainBatchProcessor(args, kg)
        elif self.args.unlearning_method == "GraphDPO":
            self.train_processor = GraphDPOTrainBatchProcessor(args, kg)
        else:
            raise NotImplementedError
        self.forget_valid_processor = ForgetDevBatchProcessor(args, kg)
        self.reserve_valid_processor = ReserveDevBatchProcessor(args, kg)

    def run_epoch(self):
        self.args.valid = True
        print("Epoch Training =========================")
        if self.args.unlearning_method == "ADVIMP":
            loss = self.train_processor.process_epoch(self.model, self.optimizer, self.origin_params)
        else:
            loss = self.train_processor.process_epoch(self.model, self.optimizer)
        if self.args.epoch % self.args.valid_gap == 0:
            print("Epoch Testing =========================")
            forget_results = self.forget_valid_processor.process_epoch(self.model)
            reserve_results = self.reserve_valid_processor.process_epoch(self.model)
            self.args.valid = False
            return loss, forget_results, reserve_results
        else:
            return loss, None, None

    def estimate_parameter_importance(self, trn_loader, model, device, num_samples, optimizer):
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(device) for n, p in model.named_parameters()
                    if p.requires_grad}
        
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        # for images, targets in itertools.islice(trn_loader, n_samples_batches):
        for h, r, t, y in itertools.islice(trn_loader, n_samples_batches):
            # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
            # outputs = model.forward(images.to(device))
            # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
            # loss = torch.norm(outputs, p=2, dim=1).mean()
            h, r, t, y = h.to(device), r.to(device), t.to(device), y.to(device)
            loss = model.loss(h, r, t, y)
            optimizer.zero_grad()
            loss.backward()
            # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
            for n, p in model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(t)
        # Eq. 2: divide by N total number of samples
        n_samples = n_samples_batches * trn_loader.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance
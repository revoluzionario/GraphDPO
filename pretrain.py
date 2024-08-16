from src.utils import *
from src.parse_args import args
from src.model.TransE import TransE
from src.data_load.KnowledgeGraph import PretrainData
from src.train import *
from src.test import *

class Instructor():
    def __init__(self, args) -> None:
        """ 1. Set parameters, seeds, logger, paths and device """
        """ Set parameters """
        self.args = args
        self.args.begin_pretrain = True
        self.args.begin_unleanring = False
        """ Set seeds """
        set_seeds(self.args.seed)
        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        now_time = get_datetime()
        self.args.log_path += now_time
        logging_file_name = f'{self.args.log_path}.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger
        """ Set paths """
        if os.path.exists(self.args.pretrain_save_path):
            shutil.rmtree(self.args.pretrain_save_path, True)
        if not os.path.exists(self.args.pretrain_save_path):
            os.mkdir(self.args.pretrain_save_path)
        """ Set device """
        torch.cuda.set_device(int(self.args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device
        print(self.args.device)
        """ 2. Define data """
        self.kg = PretrainData(self.args)
        print(len(self.kg.train_data))
        """ 3. Define model """
        self.model, self.optimizer = self.create_model()
        self.args.logger.info(args)

    def create_model(self):
        model = TransE(self.args, self.kg)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def pretrain(self):
        report_results = PrettyTable()
        report_results.field_names = ['Time', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']
        test_results = []
        training_times = []
        training_time = self.train()
        """ prepare result table """
        test_res = PrettyTable()
        test_res.field_names = [
            'MRR',
            'Hits@1',
            'Hits@3',
            'Hits@5',
            'Hits@10',
        ]

        """ save and load model """
        best_checkpoint = os.path.join(
            self.args.pretrain_save_path, f'model_best.tar'
        )
        self.load_checkpoint(best_checkpoint)

        """ predict """
        res = self.test()
        self.args.logger.info(
                    f"MRR:{round(res['mrr'] * 100, 3)}\tHits@1:{round(res['hits1'] * 100, 3)}\tHits@3:{round(res['hits3'] * 100, 3)}\tHits@10:{round(res['hits10'] * 100, 3)}"
                )

    def get_report_results(self, results):
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.train_data))
        whole_mrr = sum(
            mrr * num_test[i] for i, mrr in enumerate(mrrs)
            ) / sum(num_test)
        whole_hits1 = sum(
            hits1 * num_test[i] for i, hits1 in enumerate(hits1s)
        ) / sum(num_test)
        whole_hits3 = sum(
            hits3 * num_test[i] for i, hits3 in enumerate(hits3s)
        ) / sum(num_test)
        whole_hits10 = sum(
            hits10 * num_test[i] for i, hits10 in enumerate(hits10s)
        ) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def train(self):
        start_time = time.time()
        print("Start training =========================")
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)
        for epoch in range(int(self.args.epoch_num)):
            if self.args.debug and epoch > 0:
                break
            self.args.epoch = epoch
            loss, valid_res = trainer.run_epoch()
            if valid_res[self.args.valid_metrics] > self.best_valid:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = 0
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                self.save_model()
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info(
                        f'Early Stopping! Epoch: {epoch} Best Results: {round(self.best_valid * 100, 3)}'
                    )
                    break
            if epoch % 1 == 0:
                self.args.logger.info(
                    f"Epoch:{epoch}\tLoss:{round(loss, 3)}\tMRR:{round(valid_res['mrr'] * 100, 3)}\tHits@10:{round(valid_res['hits10'] * 100, 3)}\tBest:{round(self.best_valid * 100, 3)}"
                )
        end_time = time.time()
        return end_time - start_time

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        return tester.test()

    def save_model(self, is_best=False):
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(
            self.args.pretrain_save_path,
            f'checkpoint-{self.args.epoch}.tar',
        )
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(
                self.args.pretrain_save_path, f'model_best.tar'
            )
            shutil.copyfile(out_tar, best_path)

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info(f"=> loading checkpoint \'{input_file}\'")
            checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info(f'=> no checking found at \'{input_file}\'')


if __name__ == "__main__":
    ins = Instructor(args)
    ins.pretrain()
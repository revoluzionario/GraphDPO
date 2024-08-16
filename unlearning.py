from src.utils import *
from src.unlearning_parse_args import args
from src.model.Repretrain import Repretrain
from src.model.GraphDPO import GraphDPO
from src.data_load.KnowledgeGraph import PretrainData, UnlearningData
from src.train import *
from src.test import *

class Instructor():
    def __init__(self, args) -> None:
        """ 1. Set parameters, seeds, logger, paths and device """
        """ Set parameters """
        self.args = args

        """ Important: Set unlearning parameters """
        self.args.begin_pretrain = False
        self.args.begin_unleanring = True
        self.args.unlearning_save_path = os.path.join("./checkpoint_unlearning", self.args.unlearning_method)


        """ Set seeds """
        set_seeds(self.args.seed)
        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        now_time = get_datetime()
        self.args.log_path += now_time
        self.args.log_path += self.args.unlearning_method
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
        self.args.unlearning_save_path = self.args.unlearning_save_path + self.args.data_name
        if os.path.exists(self.args.unlearning_save_path):
            shutil.rmtree(self.args.unlearning_save_path, True)
        if not os.path.exists(self.args.unlearning_save_path):
            os.mkdir(self.args.unlearning_save_path)
        """ Set device """
        torch.cuda.set_device(int(self.args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device
        print(self.args.device)
        """ 2. Define data """
        self.kg = UnlearningData(self.args)
        print(len(self.kg.snapshots[0].forgotten_triples))
        """ 3. Define model """
        self.model, self.optimizer = self.create_model()
        self.args.logger.info(args)

    def create_model(self):
        if self.args.unlearning_method in ["repretrain", "Before", "Finetune", "NG", "RL", "Fisher", "SSD"]:
            model = Repretrain(self.args, self.kg)
        elif self.args.unlearning_method == "GraphDPO":
            model = GraphDPO(self.args, self.kg)
        else:
            model = Repretrain(self.args, self.kg)
        input_file = os.path.join(self.args.pretrain_save_path, "model_best.tar")
        checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def reset_model(self):
        model = Repretrain(self.args, self.kg)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer


    def dpo_sample(self):
        pass


    def unlearning(self):
        report_results = PrettyTable()
        report_results.field_names = ['Snapshot', 'Time', 'MRR+', 'Hits@1+', 'Hits@10+', 'MRR-', 'Hits@1-', 'Hits@10-']
        training_times = []


        # self.args.snapshot = 0
        # self.args.snapshot_test = 0
        # self.args.snapshot_valid = 0
        # forget_res, reserve_res = self.test()
        # self.args.logger.info("The performance of origin pretrained model:")
        # self.args.logger.info("Performance of forget data:")
        # self.args.logger.info(forget_res)
        # self.args.logger.info("Performace of reserve data:")
        # self.args.logger.info(reserve_res)


        for ss_id in range(int(self.args.snapshot_num)):
            self.args.snapshot = ss_id
            self.args.snapshot_test = ss_id
            self.args.snapshot_valid = ss_id

            if self.args.unlearning_method == "repretrain":
                self.model, self.optimizer = self.reset_model()

            """ Sampling """
            if self.args.unlearning_method == "GraphDPO":
                self.model.save_embeddings()
                self.dpo_sample()

            """ training """
            if self.args.unlearning_method != "Before":
                training_time = self.train()
                print(f"Snapshot {self.args.snapshot} training time is {training_time}.")
            else:
                training_time = 0.0

            """ save and load model """
            if self.args.unlearning_method != "Before":
                best_checkpoint = os.path.join(
                    self.args.unlearning_save_path, f'{str(ss_id)}model_best.tar'
                )
                self.load_checkpoint(best_checkpoint)

            """ predict """
            forget_res, reserve_res = self.test()
            print("The performance of unleanring model:")
            print("Performance of forget data:")
            print(forget_res)
            print("Performace of reserve data:")
            print(reserve_res)

            """ output """
            report_results.add_row([
                self.args.snapshot,
                training_time,
                reserve_res["mrr"],
                reserve_res["hits1"],
                reserve_res["hits10"],
                forget_res["mrr"],
                forget_res["hits1"],
                forget_res["hits10"]
            ])
            self.args.logger.info(f"\n{report_results}")
            training_times.append(training_time)

        """ final results """
        self.args.logger.info(f'Report Result:\n{report_results}')
        self.args.logger.info(f'Sum_Training_Time:{sum(training_times)}')
        self.args.logger.info(f'Every_Training_Time:{training_times}')


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
        trainer = UnlearningTrainer(self.args, self.kg, self.model, self.optimizer)
        for epoch in range(int(self.args.epoch_num)):
            if self.args.debug and epoch > 0:
                break
            self.args.epoch = epoch
            if self.args.unlearning_method in ["repretrain", "Finetune", "NG", "RL", "Fisher", "BS", "ADVIMP", "SSD", "GraphDPO"]:
                valid_res = dict()
                loss, forget_results, reserve_results = trainer.run_epoch()
                if self.args.epoch % self.args.valid_gap == 0:
                    valid_res[self.args.valid_metrics] = 0.5 * (1 - forget_results[self.args.valid_metrics]) + 0.5 * reserve_results[self.args.valid_metrics]
                    valid_res['hits3'] = 0.5 * (1 - forget_results['hits3']) + 0.5 * reserve_results['hits3']
                    valid_res['hits10'] = 0.5 * (1 - forget_results['hits10']) + 0.5 * reserve_results['hits10']
            else:
                raise NotImplementedError
            if self.args.epoch % self.args.valid_gap != 0:
                continue
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
                    f"Epoch:{epoch}\tLoss:{round(loss, 3)}\tMRR-Avg:{round(valid_res['mrr'] * 100, 3)}\tHits@10:{round(valid_res['hits10'] * 100, 3)}\tBest:{round(self.best_valid * 100, 3)}"
                )
        end_time = time.time()
        return end_time - start_time

    def test(self):
        print("Start testing =========================")
        tester = UnlearningTester(self.args, self.kg, self.model)
        return tester.test()

    def save_model(self, is_best=False):
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(
            self.args.unlearning_save_path,
            f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
        )
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(
                self.args.unlearning_save_path, f'{str(self.args.snapshot)}model_best.tar'
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
    ins.unlearning()
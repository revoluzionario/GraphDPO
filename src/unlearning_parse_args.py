import argparse

parser = argparse.ArgumentParser(
    description="Parser for Auguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# global setting
parser.add_argument("-gpu", dest="gpu", default=3, help="Device of gpu")
parser.add_argument("-data_name", dest="data_name", default="fb15k-237-20", help="Name of data")
parser.add_argument("-data_path", dest="data_path", default="./data/", help="Path of data")
parser.add_argument("-pretrain_save_path", dest="pretrain_save_path", default="./checkpoint_pretrain", help="Path of checkpoint for pretrain")
parser.add_argument("-unlearning_save_path", dest="unlearning_save_path", default="./checkpoint_unlearning", help="Path of checkpoint for unleanring")
parser.add_argument("-log_path", dest="log_path", default="./logs/", help="Path of logs")
parser.add_argument("-seed", dest="seed", default=3407, help="Set random seed")


# model setting
parser.add_argument("-snapshot_num", dest="snapshot_num", default=4, help="number of snapshot")
parser.add_argument("-unlearning_method", dest="unlearning_method", default="SSD", help="The name of unlearning")
# name list: ["repretrain", "Before", "Finetune", "NG", "RL", "Fisher", "BS", "ADVIMP", "SSD", "GraphDPO"]
parser.add_argument("-lr", dest="lr", default=1e-3, help="learning rate for models", type=float)
parser.add_argument("-begin_unleanring", dest="begin_unleanring", default=True, help="Begin unleanring or not")
parser.add_argument("-epoch_num", dest="epoch_num", default=100, help="Num of epoch")
parser.add_argument("-batch_size", dest="batch_size", default=1024, help="Batch size of data")
parser.add_argument("-debug", dest="debug", default=False, help="debug or not")
parser.add_argument("-emb_dim", dest="emb_dim", default=200, help="dim of embedding size")
parser.add_argument("-margin", dest="margin", default=8.0, help="The margin of MarginLoss")
parser.add_argument("-neg_ratio", dest="neg_ratio", default=10, help="the ratio of negtive/postive facts")
parser.add_argument("-l2", dest='l2', default=0.0, help="optimizer l2")
parser.add_argument("-patience", dest="patience", default=1, help="early stop step") # true patience = patience * valid_gap
parser.add_argument("-valid_gap", dest="valid_gap", default=3, help="valid gap")
parser.add_argument("-valid_metrics", dest="valid_metrics", default="mrr", help="Metrics for KGE")


# ablation experiments
parser.add_argument("-random_policy", dest="random_policy", default="far", help="random policy for GraphDPO")
# policy list: ["far", "close"]
parser.add_argument("-wo_dpo", dest="wo_dpo", default=False, help="without dpo")
parser.add_argument("-wo_distill", dest="wo_distill", default=False, help="without distillation")
parser.add_argument("-wo_reply", dest="wo_reply", default=False, help="without reply")

args = parser.parse_args()
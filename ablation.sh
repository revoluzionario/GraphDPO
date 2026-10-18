# python unlearning.py -unlearning_method GraphDPO -gpu 3 -data_name fb15k-237-10 -snapshot_num 4 -wo_dpo True
# python unlearning.py -unlearning_method GraphDPO -gpu 3 -data_name fb15k-237-10 -snapshot_num 4 -wo_distill True
# python unlearning.py -unlearning_method GraphDPO -gpu 3 -data_name fb15k-237-10 -snapshot_num 4 -wo_reply True
# python unlearning.py -unlearning_method GraphDPO -gpu 3 -data_name fb15k-237-10 -snapshot_num 4 -random_policy close
python unlearning.py -unlearning_method GraphDPO -gpu 1 -data_name fb15k-237-20 -snapshot_num 4 -wo_dpo True -lr 1e-2
python unlearning.py -unlearning_method GraphDPO -gpu 1 -data_name fb15k-237-20 -snapshot_num 4 -wo_distill True -lr 1e-2
# python unlearning.py -unlearning_method GraphDPO -gpu 1 -data_name fb15k-237-20 -snapshot_num 4 -wo_reply True -lr 1e-2
# python unlearning.py -unlearning_method GraphDPO -gpu 1 -data_name fb15k-237-20 -snapshot_num 4 -random_policy close -lr 1e-2
# python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-10 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -wo_dpo True
# python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-10 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -wo_distill True
# python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-10 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -wo_reply True
# python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-10 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -random_policy close
python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-20 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -wo_dpo True -lr 1e-4
python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-20 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -wo_distill True -lr 1e-4
# python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-20 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -wo_reply True -lr 1e-2
# python unlearning.py -unlearning_method GraphDPO -gpu 0 -data_name wn18rr-20 -snapshot_num 4 -pretrain_save_path ./checkpoint_pretrain -random_policy close -lr 1e-2
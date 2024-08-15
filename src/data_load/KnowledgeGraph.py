from ..utils import *


class PretrainData():
    def __init__(self, args) -> None:
        self.args = args
        self.ent_num = 0
        self.rel_num = 0
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.relationid2invid = {}
        self.hr2t = {}
        self.train_data = []
        self.load_data()

    def fact2id(self, h, r, t):
        h_id = self.entity2id[h]
        r_id = self.relation2id[r]
        t_id = self.entity2id[t]
        return (h_id, r_id, t_id)

    def load_data(self):
        train_data_dir = os.path.join(self.args.data_path, self.args.data_name)
        train_data_path = os.path.join(train_data_dir, "all_data.txt")
        with open(train_data_path, "r", encoding="utf-8") as rf:
            lines = list(rf.readlines())
            for line in lines:
                h, r, t = line.strip().split()
                if h not in self.entity2id:
                    self.entity2id[h] = self.ent_num
                    self.id2entity[self.ent_num] = h
                    self.ent_num += 1
                if r not in self.relation2id:
                    self.relation2id[r] = self.rel_num
                    self.id2relation[self.rel_num] = r
                    self.relation2id[r + "_inv"] = self.rel_num + 1
                    self.id2relation[self.rel_num + 1] = r + "_inv"
                    self.relationid2invid[self.rel_num] = self.rel_num + 1
                    self.relationid2invid[self.rel_num + 1] = self.rel_num
                    self.rel_num += 2
                if t not in self.entity2id:
                    self.entity2id[t] = self.ent_num
                    self.id2entity[self.ent_num] = t
                    self.ent_num += 1
                self.train_data.append(self.fact2id(h, r, t))
        for h, r, t in self.train_data:
            if (h, r) not in self.hr2t.keys():
                self.hr2t[(h, r)] = set()
            self.hr2t[(h, r)].add(t)
            if (t, self.relationid2invid[r], h) not in self.hr2t.keys():
                self.hr2t[(t, self.relationid2invid[r])] = set()
            self.hr2t[(t, self.relationid2invid[r])].add(h)


class UnlearningData(PretrainData):
    def __init__(self, args) -> None:
        super(UnlearningData, self).__init__(args)
        self.snapshots = {i: Snapshot(self.args) for i in range(int(self.args.snapshot_num))}
        self.load_snapshots()

    def load_snapshots(self):
        for i in range(int(self.args.snapshot_num)):
            unlearning_data_dir = os.path.join(self.args.data_path, self.args.data_name, str(i))
            unlearning_data_path = os.path.join(unlearning_data_dir, f"{i}.txt")
            with open(unlearning_data_path, "r", encoding="utf-8") as rf:
                lines = list(rf.readlines())
                for line in lines:
                    head, relation, tail, forget_entity = line.strip().split("\t")
                    is_head = True
                    if forget_entity == tail:
                        is_head = False
                    if self.args.unlearning_method == "Fisher":
                        if is_head:
                            self.snapshots[i].class_to_forget.append(self.entity2id[head])
                        else:
                            self.snapshots[i].class_to_forget.append(self.entity2id[tail])
                    self.snapshots[i].forgotten_triples.append(
                        self.fact2id(head, relation, tail)
                    )
                    self.snapshots[i].forgotten_entities.append(
                        self.entity2id[forget_entity]
                    )
                    self.snapshots[i].is_heads.append(is_head)
            accumulate_forgotten_triples = set()
            for j in range(i + 1):
                for triple in self.snapshots[j].forgotten_triples:
                    accumulate_forgotten_triples.add(triple)
            self.snapshots[i].accumulate_forgotten_triples = accumulate_forgotten_triples
            for triple in self.train_data:
                if triple not in accumulate_forgotten_triples:
                    self.snapshots[i].reserve_triples.append(triple)
            self.snapshots[i].forgotten_num = len(self.snapshots[i].forgotten_entities)
            self.snapshots[i].acc_forgotten_num = len(accumulate_forgotten_triples)
            self.snapshots[i].reserve_num = len(self.snapshots[i].reserve_triples)
            assert self.snapshots[i].acc_forgotten_num + self.snapshots[i].reserve_num == len(self.train_data)


class Snapshot():
    def __init__(self, args) -> None:
        self.args = args
        self.forgotten_triples = [] # 本次遗忘的三元组
        self.forgotten_entities = [] # 本次遗忘的实体
        self.accumulate_forgotten_triples = [] # 累计遗忘的三元组
        self.is_heads = [] # 遗忘的实体是否是头实体
        self.reserve_triples = [] # 保留三元组
        self.forgotten_num = 0 # 本次遗忘的三元组数量
        self.acc_forgotten_num = 0 # 累计遗忘的三元组数量
        self.reserve_num = 0 # 所有保留三元组的数量
        self.class_to_forget = [] # (可选:Fisher) 记录所有被遗忘的实体
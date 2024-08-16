from src.utils import *

class TrainDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(TrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.build_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def build_facts(self):
        facts = []
        for h, r, t in self.kg.train_data:
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, r + 1, h),
                'label': 1
            })
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class RepreTrainDataset(Dataset):
    """ Use retention sets for training """
    def __init__(self, args, kg) -> None:
        super(RepreTrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.build_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def build_facts(self):
        facts = []
        for h, r, t in self.kg.snapshots[self.args.snapshot].reserve_triples:
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, r + 1, h),
                'label': 1
            })
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class NGTrainDataset(Dataset):
    """ Use the forgetting set for reverse gradient propagation """
    def __init__(self, args, kg) -> None:
        super(NGTrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.build_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def build_facts(self):
        facts = []
        for h, r, t in self.kg.snapshots[self.args.snapshot].forgotten_triples:
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, r + 1, h),
                'label': 1
            })
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class RLTrainDataset(Dataset):
    """ Use random tags for forgotten sets """
    def __init__(self, args, kg) -> None:
        super(RLTrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.build_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def build_facts(self):
        facts = []
        for i, (h, r, t) in enumerate(self.kg.snapshots[self.args.snapshot].forgotten_triples):
            random_label = random.randint(0, self.kg.ent_num - 1)
            if self.kg.snapshots[self.args.snapshot].is_heads[i]:
                h = random_label
            else:
                t = random_label
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, r + 1, h),
                'label': 1
            })
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class ForgetBoundaryDataset(Dataset):
    """ Determine the forgetting boundary of the forgetting set """
    def __init__(self, args, kg) -> None:
        super(ForgetBoundaryDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.build_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def build_facts(self):
        facts = []
        linked_entities = set()
        self.args.fb_maxnum = 20000
        for i, (h, r, t) in enumerate(self.kg.snapshots[self.args.snapshot].forgotten_triples):
            if self.kg.snapshots[self.args.snapshot].is_heads[i]:
                linked_entities.add(h)
            else:
                linked_entities.add(t)
        pre_facts = []
        for i, (h, r, t) in enumerate(self.kg.snapshots[self.args.snapshot].reserve_triples):
            if h in linked_entities or t in linked_entities:
                pre_facts.append((h, r, t))
        if len(pre_facts) > self.args.fb_maxnum: # len(pre_facts): 247144
            pre_facts = random.sample(pre_facts, self.args.fb_maxnum)
        for h, r, t in pre_facts:
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, r + 1, h),
                'label': 1})
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class GraphDPOTrainDataset(Dataset):
    """ Use preference optimization for forgotten sets """
    def __init__(self, args, kg) -> None:
        super(GraphDPOTrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.entity_distill_mask = torch.zeros(self.kg.ent_num)
        self.relation_distill_mask = torch.zeros(self.kg.rel_num)
        self.facts, self.pos_facts, self.neg_facts = self.build_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        pos_fact = self.pos_facts[index]
        neg_fact = self.neg_facts[index]
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact, neg_fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, pos_fact, neg_fact

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, pos_fact, neg_fact) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        pos_fact = list(_[2] for _ in data)
        pos_fact = torch.LongTensor(pos_fact)
        neg_fact = list(_[3] for _ in data)
        neg_fact = torch.LongTensor(neg_fact)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label, pos_fact, neg_fact

    def build_facts(self):
        facts, pos_facts, neg_facts = [], [], []
        linked_entities = set()
        random_reserve_triples = deepcopy(self.kg.snapshots[self.args.snapshot].reserve_triples)
        random.shuffle(random_reserve_triples)
        random_reserve_dict = {}
        forgotten_entities = set(self.kg.snapshots[self.args.snapshot].forgotten_entities)
        for rh, rr, rt in random_reserve_triples:
            if rh not in random_reserve_dict:
                random_reserve_dict[rh] = []
            if rt not in random_reserve_dict:
                random_reserve_dict[rt] = []
            random_reserve_dict[rh].append((rh, rr, rt))
            random_reserve_dict[rt].append((rh, rr, rt))
            if rh in forgotten_entities:
                self.entity_distill_mask[rt] = 1
            if rt in forgotten_entities:
                self.entity_distill_mask[rh] = 1
        for i, (h, r, t) in enumerate(self.kg.snapshots[self.args.snapshot].forgotten_triples):
            """ Negative examples of preference optimization are identified by forgetting boundaries """
            snapshot = self.kg.snapshots[self.args.snapshot]
            is_head = snapshot.is_heads[i]
            triples_to_check = random_reserve_dict.get(h if is_head else t, [])
            random_label = random.randint(0, self.kg.ent_num - 1)
            if self.args.random_policy == "far":
                def check_label(random_label):
                    for rh, rr, rt in triples_to_check:
                        if rh == random_label or rt == random_label:
                            return True
                        return False
                MAX_ITER_NUM = 10
                iter_num = 0
                while(check_label(random_label)):
                    random_label = random.randint(0, self.kg.ent_num - 1)
                    iter_num += 1
                    if iter_num > MAX_ITER_NUM:
                        break
            elif self.args.random_policy == "close":
                random_label = None
                for rh, rr, rt in triples_to_check:
                    if is_head and (h == rh or h == rt):
                        random_label = (rt if h == rh else rh)
                        break
                    elif not is_head and (t == rh or t == rt):
                        random_label = (rt if t == rh else rh)
                        break
                if random_label == None:
                    random_label = random.randint(0, self.kg.ent_num - 1)
            if self.kg.snapshots[self.args.snapshot].is_heads[i]:
                linked_entities.add(h)
                pos_facts.append((random_label, r, t))
                pos_facts.append((t, r + 1, random_label))
                facts.append({
                    'fact': (random_label, r, t),
                    'label': 1})
                facts.append({
                    'fact': (t, r + 1, random_label),
                    'label': 1})
            else:
                linked_entities.add(t)
                pos_facts.append((h, r, random_label))
                pos_facts.append((random_label, r + 1, h))
                facts.append({
                    'fact': (h, r, random_label),
                    'label': 1})
                facts.append({
                    'fact': (random_label, r + 1, h),
                    'label': 1})
            neg_facts.append((h, r, t))
            neg_facts.append((t, r + 1, h))
        return facts, pos_facts, neg_facts

    def corrupt(self, fact, neg_facts):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 9 neg + 1 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio - 1)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio - 1)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio - 1)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        facts.append((neg_facts[0], neg_facts[1], neg_facts[2]))
        label.append(-1)
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class TestDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(TestDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.test = self.build_facts()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, index):
        ele = self.test[index]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        """ for valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        y = np.zeros([self.kg.ent_num], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        test = []
        for h, r, t in self.kg.train_data:
            test.append({
                'fact': (h, r, t),
                'label': self.kg.hr2t[(h, r)]
            })
            test.append({
                'fact': (t, r + 1, h),
                'label': self.kg.hr2t[(t, r + 1)]
            })
        return test

class ForgetTestDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(ForgetTestDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.valid, self.test = self.build_facts()

    def __len__(self):
        if self.args.valid:
            return len(self.valid)
        else:
            return len(self.test)

    def __getitem__(self, index):
        if self.args.valid:
            ele = self.valid[index]
        else:
            ele = self.test[index]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        """ for valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        y = np.zeros([self.kg.ent_num], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        valid, test = [], []
        for h, r, t in self.kg.snapshots[self.args.snapshot_test].accumulate_forgotten_triples:
            test.append({
                'fact': (h, r, t),
                'label': self.kg.hr2t[(h, r)]
            })
            test.append({
                'fact': (t, r + 1, h),
                'label': self.kg.hr2t[(t, r + 1)]
            })
        for h, r, t in self.kg.snapshots[self.args.snapshot_valid].accumulate_forgotten_triples:
            valid.append({
                'fact': (h, r, t),
                'label': self.kg.hr2t[(h, r)]
            })
            valid.append({
                'fact': (t, r + 1, h),
                'label': self.kg.hr2t[(t, r + 1)]
            })
        return valid, test


class ReserveTestDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(ReserveTestDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.valid, self.test = self.build_facts()

    def __len__(self):
        if self.args.valid:
            return len(self.valid)
        else:
            return len(self.test)

    def __getitem__(self, index):
        if self.args.valid:
            ele = self.valid[index]
        else:
            ele = self.test[index]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        """ for valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        y = np.zeros([self.kg.ent_num], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        valid, test = [], []
        for h, r, t in self.kg.snapshots[self.args.snapshot_test].reserve_triples:
            test.append({
                'fact': (h, r, t),
                'label': self.kg.hr2t[(h, r)]
            })
            test.append({
                'fact': (t, r + 1, h),
                'label': self.kg.hr2t[(t, r + 1)]
            })
        for h, r, t in self.kg.snapshots[self.args.snapshot_valid].reserve_triples:
            valid.append({
                'fact': (h, r, t),
                'label': self.kg.hr2t[(h, r)]
            })
            valid.append({
                'fact': (t, r + 1, h),
                'label': self.kg.hr2t[(t, r + 1)]
            })
        return valid, test
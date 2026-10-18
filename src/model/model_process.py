from ..utils import *
from ..data_load.data_loader import *

class TrainBatchProcessor():
    """ Normal training process """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        if self.args.begin_pretrain:
            self.dataset = TrainDataset(args, kg)
        elif self.args.begin_unleanring:
            if self.args.unlearning_method in ["repretrain", "Finetune"]:
                self.dataset = RepreTrainDataset(args, kg)
        else:
            raise ImportError
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )

    def process_epoch(self, model, optimizer):
        model.train()
        """ Start training """
        total_loss = 0.0
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class NGTrainBatchProcessor():
    """ Training process using inverse gradient ascension """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = NGTrainDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )

    def process_epoch(self, model, optimizer):
        model.train()
        """ Start training """
        total_loss = 0.0
        if self.args.data_name == "fb15k-237-10" or self.args.data_name == "fb15k-237-20":
            self.args.lr = 1
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            batch_loss.backward()
            # optimizer.step()
            with torch.no_grad():
                for param in model.parameters():
                    param += param.grad * self.args.lr
            total_loss += batch_loss.item()
        return total_loss


class RLTrainBatchProcessor():
    """ Training process using random labels """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = RLTrainDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )

    def process_epoch(self, model, optimizer):
        model.train()
        """ Start training """
        total_loss = 0.0
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
        args=None
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None
        self.args = args

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary


    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            bh, br, bt, by = batch
            self.opt.zero_grad()
            batch_loss = self.model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            batch_loss.backward()
            # x, _, y = batch
            # x, y = x.to(self.device), y.to(self.device)
            # self.opt.zero_grad()
            # out = self.model(x)
            # loss = criterion(out, y)
            # loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances


    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)


class SSDTrainBatchProcessor():
    """
    SSD: From the original codes
    """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.forget_dataset = RLTrainDataset(args, kg)
        self.reserve_dataset = RepreTrainDataset(args, kg)
        self.forget_data_loader = DataLoader(
            self.forget_dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.forget_dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )
        self.reserve_data_loader = DataLoader(
            self.reserve_dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.reserve_dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )

    def process_epoch(self, model, optimizer):
        model.train()
        """ Start training """
        total_loss = 0.0
        dampening_constant = 1 # from the paper
        selection_weighting = 1 # from the paper
        parameters = {
            "lower_bound": 1,  # unused
            "exponent": 1,  # unused
            "magnitude_diff": None,  # unused
            "min_layer": -1,  # -1: all layers are available for modification
            "max_layer": -1,  # -1: all layers are available for modification
            "forget_threshold": 1,  # unused
            "dampening_constant": dampening_constant,  # Lambda from paper
            "selection_weighting": selection_weighting,  # Alpha from paper
        }
        # load the trained model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pdr = ParameterPerturber(model, optimizer, self.args.device, parameters, self.args)
        model = model.eval()
        # Calculation of the forget set importances
        sample_importances = pdr.calc_importance(self.forget_data_loader)
        # print("okkk")
        # exit()
        # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
        original_importances = pdr.calc_importance(self.reserve_data_loader)
        # Dampen selected parameters
        pdr.modify_weight(original_importances, sample_importances)
        # for b_id, batch in enumerate(self.data_loader):
        #     bh, br, bt, by = batch
        #     optimizer.zero_grad()
        #     batch_loss = model.loss(bh.to(self.args.device),
        #                             br.to(self.args.device),
        #                             bt.to(self.args.device),
        #                             by.to(self.args.device) if by is not None else by).float()
        #     batch_loss.backward()
        #     optimizer.step()
        #     total_loss += batch_loss.item()
        return total_loss


class GraphDPOTrainBatchProcessor():
    """ dpo """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = GraphDPOTrainDataset(args, kg)
        self.entity_distill_mask = self.dataset.entity_distill_mask
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )
        self.forget_boundary_dataset = ForgetBoundaryDataset(args, kg)
        self.forget_boundary_dataloader = DataLoader(
            self.forget_boundary_dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.forget_boundary_dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )

    def process_epoch(self, model, optimizer):
        model.entity_distill_mask = self.entity_distill_mask
        model.train()
        """ Start training """
        total_loss = 0.0
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by, pos_facts, neg_facts = batch # bh:11264 or 12288(11x or 12x) neg_facts:1024
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            if not self.args.wo_dpo:
                batch_loss += model.dpo_loss(pos_facts.to(self.args.device), neg_facts.to(self.args.device)) # pos_facts: [1024, 3]
                print(batch_loss)
            if not self.args.wo_distill:
                batch_loss += model.distill_loss()
                print(batch_loss)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        if not self.args.wo_reply:
            for b_id, batch in enumerate(self.forget_boundary_dataloader):
                bh, br, bt, by = batch
                optimizer.zero_grad()
                batch_loss = model.loss(bh.to(self.args.device),
                                        br.to(self.args.device),
                                        bt.to(self.args.device),
                                        by.to(self.args.device) if by is not None else by).float()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
        return total_loss


class ADVIMPTrainBatchProcessor():
    """
    ADV+IMP: From the original codes
    """
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.dataset = RLTrainDataset(args, kg)
        self.importance = None
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=1,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )
        self.args.pgd_eps = 2.0
        self.args.pgd_alpha = 0.1
        self.args.pgd_iter = 100
        adversary = L2PGDAttack(self.model, eps=self.args.pgd_eps, eps_iter=self.args.pgd_alpha, nb_iter=self.args.pgd_iter,
                                rand_init=True, targeted=True)
        self.adv_attack(self.model, self.args.device, self.data_loader, adversary, self.kg.ent_num)

    def adv_attack(self, model, device, train_loader, adversary, num_adv_images = None):
        model.eval()
        attacked_image_arr = []
        target_label_arr = []
        MAX_BATCH_NUM = 10
        for batch_idx, (h, r, t, y) in tqdm(enumerate(train_loader)):
            if batch_idx > MAX_BATCH_NUM:
                break
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)
            y = y.to(device)
            attack_label = torch.rand(h.shape[0], device=self.args.device) * 10
            attack_label = attack_label.to(torch.long)
            fallback_labels = (torch.rand(h.shape[0], device=self.args.device).long() * 10 + 10) // 2
            attack_label = torch.where(attack_label == t, fallback_labels, attack_label)
            h_embeds = torch.index_select(self.model.ent_embeddings.weight, 0, h)
            r_embeds = torch.index_select(self.model.rel_embeddings.weight, 0, r)
            input_embeds = torch.cat([h_embeds, r_embeds], 0)
            adv_example = adversary.perturb(input_embeds, attack_label) # [22, 200]
            n = adv_example.size(0)
            half_n = n // 2
            h_adv, r_adv = adv_example[:half_n], adv_example[half_n:] # [11, 200], [11, 200]
            with torch.no_grad():
                self.model.ent_embeddings.weight[h] = h_adv
                self.model.rel_embeddings.weight[r] = r_adv
        return None

    def process_epoch(self, model, optimizer, origin_params):
        model.train()
        """ Start training """
        total_loss = 0.0
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            for n, p in model.named_parameters():
                if n in self.importance.keys():
                    batch_loss += torch.sum(self.importance[n] * (p - origin_params[n]).pow(2)) / 2
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class AttackBase(object):
    def __init__(self, model=None, norm=False, discrete=True, device=None):
        self.model = model
        self.norm = norm
        if self.norm:
            self.mean = torch.tensor([0.4914, 0.4822, 0.2265]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss()

    def loss(self, custom_loss=None):
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(self.device)

    def perturb(self, x):
        raise NotImplementedError

    def normalize(self, x):
        if self.norm:
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def inverse_normalize(self, x):
        if self.norm:
            return x * self.std.to(x.device) + self.mean.to(x.device)
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255

    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric)
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)


class FGSM(AttackBase):
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, **kwargs):
        super(FGSM, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.rand = random_start

    def perturb(self, h, r, t, y, model=None, bound=None, device=None, **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(h.detach().clone().to(device))
        x_adv = h.detach().clone().to(device)
        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, x_nat, bound=bound,
                                 inverse_normalized=True)
            if self.discrete:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        # pred = model.predict(h, r)
        # if criterion.__class__.__name__ == "NLLLoss":
        #     pred = F.softmax(pred, dim=-1)
        loss = model.loss(h, r, t, y)
        loss.backward()

        try:
            grad_sign = x_adv.grad.data.detach().sign()
        except:
            grad_sign = 0
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()


class BSTrainBatchProcessor():
    """
    Boundary Shrink
    """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = RLTrainDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=True
        )

    def PM(self, logit, target):
        eye = torch.eye(logit.shape[1], device=logit.device)
        probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
        top2_probs = logit.softmax(1).topk(2, largest=True)
        GT_in_top2_ind = (top2_probs[1] == target.view(-1, 1)).float().sum(1) == 1
        probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:, 0]).detach()
        return probs_2nd - probs_GT

    def weight_assign(self, logit, target, bias, slope):
        pm = self.PM(logit, target)
        reweight = ((pm + bias) * slope).sigmoid().detach()
        normalized_reweight = reweight * 3
        return normalized_reweight

    def _find_z(self, model, inputs, targets, h):
        inputs.requires_grad_()
        outputs = model(inputs)
        loss_z = nn.CrossEntropyLoss()(model(inputs), targets)
        loss_z.backward()
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.0
        z = 1. * (h) * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
        inputs.grad.zero_()
        model.zero_grad()
        return z, norm_grad

    def curvature(self, model, inputs, targets, h=3., lambda_=4):
        z, norm_grad = self._find_z(model, inputs, targets, h)
        inputs.requires_grad_()
        outputs_pos = model(inputs + z)
        outputs_orig = model(inputs)
        loss_pos = nn.CrossEntropyLoss()(outputs_pos, targets)
        loss_orig = nn.CrossEntropyLoss()(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs, create_graph=True)[0]
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        model.zero_grad()
        return torch.sum(lambda_ * reg) / float(inputs.size(0)), reg

    def inf_generator(self, iterable):
        iterator = iterable.__iter__()
        while True:
            try:
                yield iterator.__next__()
            except StopIteration:
                iterator = iterable.__iter__()

    def process_epoch(self, model, optimizer):
        model.train()
        self.args.poison_epoch = 10
        total_loss = 0.0
        norm = True
        random_start = False
        test_model = deepcopy(model).to(self.args.device)
        unlearn_model = model.to(self.args.device)
        adv = FGSM(test_model, bound=0.1, norm=norm, random_start=random_start, device=self.args.device)
        criterion = nn.CrossEntropyLoss()
        forget_data_gen = self.inf_generator(self.data_loader)
        batches_per_epoch = len(self.data_loader)
        num_hits = 0
        num_sum = 0
        nearest_label = []
        for itr in tqdm(range(self.args.poison_epoch * batches_per_epoch)):
            h, r, t, y = next(forget_data_gen)
            h = h.to(self.args.device)
            r = r.to(self.args.device)
            t = t.to(self.args.device)
            y = y.to(self.args.device)
            num_sum += t.shape[0]
            # adv_train
            unlearn_model.train()
            unlearn_model.zero_grad()
            optimizer.zero_grad()
            ori_loss = unlearn_model.loss(h, r, t, y)
            loss = ori_loss
            loss.backward()
            optimizer.step()
        try:
            print('attack success ratio:', (num_hits / num_sum).float())
        except:
            pass
        # torch.save(unlearn_model, '{}boundary_shrink_unlearn_model.pth'.format(self.args.path))
        return total_loss


class FisherTrainBatchProcessor():
    """
    Fisher: From the original codes
    """
    # def __init__(self, args, kg) -> None:
    #     self.args = args
    #     self.kg = kg
    #     self.dataset = RepreTrainDataset(args, kg)
    #     self.data_loader = DataLoader(
    #         self.dataset,
    #         shuffle=True,
    #         batch_size=1,
    #         collate_fn=self.dataset.collate_fn,
    #         generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
    #         pin_memory=True
    #     )

    # def hessian(self, dataset, model):
    #     model.eval()
    #     train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #     loss_fn = nn.CrossEntropyLoss()
    #     for p in model.parameters():
    #         p.grad_acc = 0
    #         p.grad2_acc = 0
    #     MAX_TRAIN_STEP = 3
    #     for b_id, batch in tqdm(enumerate(self.data_loader)):
    #         if b_id > MAX_TRAIN_STEP:
    #             break
    #         bh, br, bt, by = batch
    #         bh = bh.to(self.args.device)
    #         br = br.to(self.args.device)
    #         bt = bt.to(self.args.device)
    #         by = by.to(self.args.device) # -1 or 1
    #         output = model.predict(bh, br) # [11, 145541]
    #         prob = F.softmax(output, dim=-1) # [11, 145541]
    #         output_new = torch.argmax(output, dim=1) # [11]
    #         for y in range(output.shape[1]):
    #             target = torch.empty_like(bt).fill_(y)
    #             loss = model.loss(bh, br, target, by if by is not None else by).float()
    #             model.zero_grad()
    #             loss.backward(retain_graph=True)
    #             for p in model.parameters():
    #                 if p.requires_grad:
    #                     p.grad_acc += (bt == target).sum().float() * p.grad.data
    #                     # p.grad2_acc += prob[:, y].sum().float() * p.grad.data.pow(2)
    #                     p.grad2_acc += prob[:, y].sum().float() * p.grad.data * p.grad.data
    #     for p in model.parameters():
    #         p.grad_acc /= len(train_loader)
    #         p.grad2_acc /= len(train_loader)


    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = RepreTrainDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=1,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)), # use seed generator
            pin_memory=True
        )
        self.scaler = GradScaler()

    def hessian(self, dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()
        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0
        MAX_TRAIN_STEP = 3
        accumulation_steps = 2
        for b_id, batch in tqdm(enumerate(self.data_loader)):
            if b_id > MAX_TRAIN_STEP:
                break
            bh, br, bt, by = batch
            bh = bh.to(self.args.device)
            br = br.to(self.args.device)
            bt = bt.to(self.args.device)
            by = by.to(self.args.device) # -1 or 1
            with autocast():
                output = model.predict(bh, br) # [11, 145541]
                prob = F.softmax(output, dim=-1) # [11, 145541]
                output_new = torch.argmax(output, dim=1) # [11]
                for y in range(output.shape[1]):
                    target = torch.empty_like(bt).fill_(y)
                    loss = model.loss(bh, br, target, by if by is not None else by).float()
                    model.zero_grad()
                    self.scaler.scale(loss).backward(retain_graph=True)
                    if (b_id + 1) % accumulation_steps == 0:
                        for p in model.parameters():
                            if p.requires_grad:
                                p.grad_acc += (bt == target).sum().float() * p.grad.data
                                # p.grad2_acc += prob[:, y].sum().float() * p.grad.data.pow(2)
                                p.grad2_acc += prob[:, y].sum().float() * p.grad.data * p.grad.data
                        model.zero_grad()
        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)


    def get_mean_var(self, p, is_base_dist=False, alpha=3e-6):
        var = 1./(p.grad2_acc+1e-8).clone()
        var = var.clamp(max=1e3)
        if p.size(0) == self.kg.ent_num:
            var = var.clamp(max=1e2)
        var = alpha * var
        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = p.data0.clone()
        else:
            mu = p.data0.clone()
        if p.size(0) == self.kg.ent_num and self.kg.snapshots[self.args.snapshot].forgotten_num is None:
            mu[self.kg.snapshots[self.args.snapshot].class_to_forget] = 0
            var[self.kg.snapshots[self.args.snapshot].class_to_forget] = 0.0001
        if p.size(0) == self.kg.ent_num:
            # Last layer
            var *= 10
        elif p.ndim == 1:
            # BatchNorm
            var *= 10
    #         var*=1
        return mu, var


    def kl_divergence_fisher(self, mu0, var0, mu1, var1):
        return ((mu1 - mu0).pow(2)/var0 + var1/var0 - torch.log(var1/var0) - 1).sum()


    def process_epoch(self, model, optimizer):
        """ Start training """
        model0 = deepcopy(model)
        for p in model0.parameters():
            p.data0 = p.data.clone()
        modelf = model
        for p in itertools.chain(modelf.parameters()):
            p.data0 = deepcopy(p.data.clone())
        self.hessian(self.dataset, modelf)
        self.hessian(self.dataset, model0)
        # Computes the amount of information not forgotten at all layers using the given alpha
        alpha = 1e-6
        total_kl = 0
        for (k, p), (k0, p0) in zip(modelf.named_parameters(), model0.named_parameters()):
            mu0, var0 = self.get_mean_var(p, False, alpha=alpha)
            mu1, var1 = self.get_mean_var(p0, True, alpha=alpha)
            kl = self.kl_divergence_fisher(mu0, var0, mu1, var1).item()
            total_kl += kl
            print(k, f'{kl:.1f}')
        print("Total:", total_kl)
        fisher_dir = []
        alpha = 1e-6
        for i, p in enumerate(modelf.parameters()):
            mu, var = self.get_mean_var(p, False, alpha=alpha)
            p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
            fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
        for i, p in enumerate(model0.parameters()):
            mu, var = self.get_mean_var(p, False, alpha=alpha)
            p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        return 0.0


class DevBatchProcessor():
    """
    test
    """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=True
        )

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = {}
        for batch in self.data_loader:
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)
            num += len(head)
            pred = model.predict(head, relation)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            """ rank all candidate entities """
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in [0, 2, 9]:
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results



class ForgetDevBatchProcessor():
    """
    test for forgetting set
    """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        self.dataset = ForgetTestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=True
        )

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = {}
        for batch in self.data_loader:
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)
            num += len(head)
            pred = model.predict(head, relation)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            """ rank all candidate entities """
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in [0, 2, 9]:
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results



class ReserveDevBatchProcessor():
    """
    test for reservation
    """
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        self.dataset = ReserveTestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=True
        )

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = {}
        for batch in self.data_loader:
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)
            num += len(head)
            pred = model.predict(head, relation)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            """ rank all candidate entities """
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in [0, 2, 9]:
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results
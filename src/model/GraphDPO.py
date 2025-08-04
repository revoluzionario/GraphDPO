from ..utils import *
from src.model.TransE import TransE
from src.model.ComplEx import ComplEx
from src.model.SimplEx import SimplEx
from src.model.RotatE import RotatE


class GraphDPO(TransE):
    def __init__(self, args, kg) -> None:
        super(GraphDPO, self).__init__(args, kg)
        self.args = args
        self.kg = kg
        self.ent_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None

    def save_embeddings(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())
        # for name, value in self.named_buffers():
        #     print(name)

    def embedding(self):
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def old_embeddings(self):
        for name, value in self.named_buffers():
            if name == "old_data_ent_embeddings_weight":
                old_data_ent_embeddings_weight = value
            elif name == "old_data_rel_embeddings_weight":
                old_data_rel_embeddings_weight = value
        return old_data_ent_embeddings_weight, old_data_rel_embeddings_weight

    def ent_norm(self, e):
        return F.normalize(e, 2, -1)

    def rel_norm(self, r):
        return F.normalize(r, 2, -1)

    def score_fun(self, h, r, t):
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t = self.ent_norm(t)
        return torch.norm(h + r - t, 1, -1)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def margin_loss(self, head, relation, tail, label=None):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.margin_loss(head, relation, tail, label) / head.size(0)
        return loss

    def distill_loss(self):
        ent_embeddings, rel_embeddings = self.embedding()
        self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)
        ent_embeddings = ent_embeddings * self.entity_distill_mask.unsqueeze(1)
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()
        old_data_ent_embeddings = old_data_ent_embeddings * self.entity_distill_mask.unsqueeze(1)
        ent_distill_loss = self.huber_loss(ent_embeddings, old_data_ent_embeddings)
        rel_distill_loss = self.huber_loss(rel_embeddings, old_data_rel_embeddings)
        distill_loss = ent_distill_loss + rel_distill_loss
        self.args.distill_lambda = 0.001
        distill_loss = self.args.distill_lambda * distill_loss
        return distill_loss

    def dpo_loss(self, pos_triples, neg_triples):
        ent_embeddings, rel_embeddings = self.embedding()
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()
        pos_head, pos_relation, pos_tail = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        neg_head, neg_relation, neg_tail = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        new_pos_h = torch.index_select(ent_embeddings, 0, pos_head)
        new_pos_r = torch.index_select(rel_embeddings, 0, pos_relation)
        new_pos_t = torch.index_select(ent_embeddings, 0, pos_tail)
        new_neg_h = torch.index_select(ent_embeddings, 0, neg_head)
        new_neg_r = torch.index_select(rel_embeddings, 0, neg_relation)
        new_neg_t = torch.index_select(ent_embeddings, 0, neg_tail)
        new_pos_scores = self.score_fun(new_pos_h, new_pos_r, new_pos_t)
        new_neg_scores = self.score_fun(new_neg_h, new_neg_r, new_neg_t)
        old_pos_h = torch.index_select(old_data_ent_embeddings, 0, pos_head)
        old_pos_r = torch.index_select(old_data_rel_embeddings, 0, pos_relation)
        old_pos_t = torch.index_select(old_data_ent_embeddings, 0, pos_tail)
        old_neg_h = torch.index_select(old_data_ent_embeddings, 0, neg_head)
        old_neg_r = torch.index_select(old_data_rel_embeddings, 0, neg_relation)
        old_neg_t = torch.index_select(old_data_ent_embeddings, 0, neg_tail)
        old_pos_scores = self.score_fun(old_pos_h, old_pos_r, old_pos_t)
        old_neg_scores = self.score_fun(old_neg_h, old_neg_r, old_neg_t)
        log_softmax_new_pos = F.log_softmax(new_pos_scores, dim=-1)
        log_softmax_old_pos = F.log_softmax(old_pos_scores, dim=-1)
        log_softmax_new_neg = F.log_softmax(new_neg_scores, dim=-1)
        log_softmax_old_neg = F.log_softmax(old_neg_scores, dim=-1)
        self.args.dpo_beta = 1
        difference = (log_softmax_new_pos - log_softmax_old_pos) - (log_softmax_new_neg - log_softmax_old_neg)
        loss = -F.log_softmax(self.args.dpo_beta * difference, dim=-1)
        loss = torch.mean(loss)
        self.args.dpo_lamba = 0.1
        loss = self.args.dpo_lamba * loss
        # loss = F.log_softmax(self.args.dpo_beta * [(log_softmax_new_pos - log_softmax_old_pos) - (log_softmax_new_neg- log_softmax_old_neg)], dim=-1)
        return loss


    def predict(self, head, relation):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t_all = ent_embeddings
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t_all = self.ent_norm(t_all)

        """ h + r - t """
        pred_t = h + r
        score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)
        score = torch.sigmoid(score)
        return score
    

class ComplExDPO(ComplEx):
    def __init__(self, args, kg) -> None:
        super(ComplExDPO, self).__init__(args, kg)
        self.args = args
        self.kg = kg
        # ComplEx uses complex embeddings, so we need 2x dimensions
        self.ent_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim * 2).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim * 2).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None

    def save_embeddings(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def embedding(self):
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def old_embeddings(self):
        for name, value in self.named_buffers():
            if name == "old_data_ent_embeddings_weight":
                old_data_ent_embeddings_weight = value
            elif name == "old_data_rel_embeddings_weight":
                old_data_rel_embeddings_weight = value
        return old_data_ent_embeddings_weight, old_data_rel_embeddings_weight

    def get_complex_embeddings(self, embeddings):
        """Split embeddings into real and imaginary parts"""
        real = embeddings[:, :self.args.emb_dim]
        imag = embeddings[:, self.args.emb_dim:]
        return real, imag

    def complex_multiply(self, h_real, h_imag, r_real, r_imag):
        """Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i"""
        real = h_real * r_real - h_imag * r_imag
        imag = h_real * r_imag + h_imag * r_real
        return real, imag

    def complex_conjugate(self, real, imag):
        """Complex conjugate: (a + bi)* = (a - bi)"""
        return real, -imag

    def score_fun(self, h, r, t):
        h_real, h_imag = self.get_complex_embeddings(h)
        r_real, r_imag = self.get_complex_embeddings(r)
        t_real, t_imag = self.get_complex_embeddings(t)
        
        # h * r * conj(t)
        hr_real, hr_imag = self.complex_multiply(h_real, h_imag, r_real, r_imag)
        t_conj_real, t_conj_imag = self.complex_conjugate(t_real, t_imag)
        
        # Final multiplication
        score_real, score_imag = self.complex_multiply(hr_real, hr_imag, t_conj_real, t_conj_imag)
        
        # Return real part as score
        return -torch.sum(score_real, dim=-1)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def margin_loss(self, head, relation, tail, label=None):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.margin_loss(head, relation, tail, label) / head.size(0)
        return loss

    def distill_loss(self):
        ent_embeddings, rel_embeddings = self.embedding()
        self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)
        ent_embeddings = ent_embeddings * self.entity_distill_mask.unsqueeze(1)
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()
        old_data_ent_embeddings = old_data_ent_embeddings * self.entity_distill_mask.unsqueeze(1)
        ent_distill_loss = self.huber_loss(ent_embeddings, old_data_ent_embeddings)
        rel_distill_loss = self.huber_loss(rel_embeddings, old_data_rel_embeddings)
        distill_loss = ent_distill_loss + rel_distill_loss
        self.args.distill_lambda = 0.001
        distill_loss = self.args.distill_lambda * distill_loss
        return distill_loss

    def dpo_loss(self, pos_triples, neg_triples):
        ent_embeddings, rel_embeddings = self.embedding()
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()
        pos_head, pos_relation, pos_tail = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        neg_head, neg_relation, neg_tail = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        
        # New model scores
        new_pos_h = torch.index_select(ent_embeddings, 0, pos_head)
        new_pos_r = torch.index_select(rel_embeddings, 0, pos_relation)
        new_pos_t = torch.index_select(ent_embeddings, 0, pos_tail)
        new_neg_h = torch.index_select(ent_embeddings, 0, neg_head)
        new_neg_r = torch.index_select(rel_embeddings, 0, neg_relation)
        new_neg_t = torch.index_select(ent_embeddings, 0, neg_tail)
        new_pos_scores = self.score_fun(new_pos_h, new_pos_r, new_pos_t)
        new_neg_scores = self.score_fun(new_neg_h, new_neg_r, new_neg_t)
        
        # Old model scores
        old_pos_h = torch.index_select(old_data_ent_embeddings, 0, pos_head)
        old_pos_r = torch.index_select(old_data_rel_embeddings, 0, pos_relation)
        old_pos_t = torch.index_select(old_data_ent_embeddings, 0, pos_tail)
        old_neg_h = torch.index_select(old_data_ent_embeddings, 0, neg_head)
        old_neg_r = torch.index_select(old_data_rel_embeddings, 0, neg_relation)
        old_neg_t = torch.index_select(old_data_ent_embeddings, 0, neg_tail)
        old_pos_scores = self.score_fun(old_pos_h, old_pos_r, old_pos_t)
        old_neg_scores = self.score_fun(old_neg_h, old_neg_r, old_neg_t)
        
        # DPO loss computation
        log_softmax_new_pos = F.log_softmax(new_pos_scores, dim=-1)
        log_softmax_old_pos = F.log_softmax(old_pos_scores, dim=-1)
        log_softmax_new_neg = F.log_softmax(new_neg_scores, dim=-1)
        log_softmax_old_neg = F.log_softmax(old_neg_scores, dim=-1)
        
        self.args.dpo_beta = 1
        difference = (log_softmax_new_pos - log_softmax_old_pos) - (log_softmax_new_neg - log_softmax_old_neg)
        loss = -F.log_softmax(self.args.dpo_beta * difference, dim=-1)
        loss = torch.mean(loss)
        self.args.dpo_lamba = 0.1
        loss = self.args.dpo_lamba * loss
        return loss

    def predict(self, head, relation):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t_all = ent_embeddings
        
        h_real, h_imag = self.get_complex_embeddings(h)
        r_real, r_imag = self.get_complex_embeddings(r)
        t_all_real, t_all_imag = self.get_complex_embeddings(t_all)
        
        # h * r
        hr_real, hr_imag = self.complex_multiply(h_real, h_imag, r_real, r_imag)
        
        # Expand for all entities
        hr_real = hr_real.unsqueeze(1)
        hr_imag = hr_imag.unsqueeze(1)
        
        # conj(t_all)
        t_all_conj_real, t_all_conj_imag = self.complex_conjugate(t_all_real, t_all_imag)
        
        # hr * conj(t_all)
        score_real, score_imag = self.complex_multiply(
            hr_real, hr_imag, 
            t_all_conj_real.unsqueeze(0), t_all_conj_imag.unsqueeze(0)
        )
        
        score = torch.sum(score_real, dim=-1)
        score = torch.sigmoid(score)
        return score
    

class SimplExDPO(SimplEx):
    def __init__(self, args, kg) -> None:
        super(SimplExDPO, self).__init__(args, kg)
        self.args = args
        self.kg = kg
        # SimplEx uses head, relation and tail specific embeddings
        self.ent_h_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim).to(self.args.device).double()
        self.ent_t_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim).to(self.args.device).double()
        self.rel_inv_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim).to(self.args.device).double()
        
        xavier_normal_(self.ent_h_embeddings.weight)
        xavier_normal_(self.ent_t_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)
        xavier_normal_(self.rel_inv_embeddings.weight)
        
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None

    def save_embeddings(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def embedding(self):
        return (self.ent_h_embeddings.weight, self.ent_t_embeddings.weight, 
                self.rel_embeddings.weight, self.rel_inv_embeddings.weight)

    def old_embeddings(self):
        old_data_ent_h_embeddings_weight = None
        old_data_ent_t_embeddings_weight = None
        old_data_rel_embeddings_weight = None
        old_data_rel_inv_embeddings_weight = None
        
        for name, value in self.named_buffers():
            if name == "old_data_ent_h_embeddings_weight":
                old_data_ent_h_embeddings_weight = value
            elif name == "old_data_ent_t_embeddings_weight":
                old_data_ent_t_embeddings_weight = value
            elif name == "old_data_rel_embeddings_weight":
                old_data_rel_embeddings_weight = value
            elif name == "old_data_rel_inv_embeddings_weight":
                old_data_rel_inv_embeddings_weight = value
        
        return (old_data_ent_h_embeddings_weight, old_data_ent_t_embeddings_weight,
                old_data_rel_embeddings_weight, old_data_rel_inv_embeddings_weight)

    def score_fun(self, h, r, t, h_emb=None, t_emb=None, r_emb=None, r_inv_emb=None):
        if h_emb is None:
            h_emb = self.ent_h_embeddings(h)
        if t_emb is None:
            t_emb = self.ent_t_embeddings(t)
        if r_emb is None:
            r_emb = self.rel_embeddings(r)
        if r_inv_emb is None:
            r_inv_emb = self.rel_inv_embeddings(r)
        
        # SimplEx score: <h_h, r> + <t_t, r_inv>
        score_h = torch.sum(h_emb * r_emb, dim=-1)
        score_t = torch.sum(t_emb * r_inv_emb, dim=-1)
        
        return -(score_h + score_t)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def margin_loss(self, head, relation, tail, label=None):
        ent_h_emb, ent_t_emb, rel_emb, rel_inv_emb = self.embedding()
        h_emb = torch.index_select(ent_h_emb, 0, head)
        t_emb = torch.index_select(ent_t_emb, 0, tail)
        r_emb = torch.index_select(rel_emb, 0, relation)
        r_inv_emb = torch.index_select(rel_inv_emb, 0, relation)
        
        score = self.score_fun(head, relation, tail, h_emb, t_emb, r_emb, r_inv_emb)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.margin_loss(head, relation, tail, label) / head.size(0)
        return loss

    def distill_loss(self):
        ent_h_emb, ent_t_emb, rel_emb, rel_inv_emb = self.embedding()
        self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)
        
        # Apply mask to entity embeddings
        ent_h_emb = ent_h_emb * self.entity_distill_mask.unsqueeze(1)
        ent_t_emb = ent_t_emb * self.entity_distill_mask.unsqueeze(1)
        
        old_ent_h_emb, old_ent_t_emb, old_rel_emb, old_rel_inv_emb = self.old_embeddings()
        old_ent_h_emb = old_ent_h_emb * self.entity_distill_mask.unsqueeze(1)
        old_ent_t_emb = old_ent_t_emb * self.entity_distill_mask.unsqueeze(1)
        
        ent_h_distill_loss = self.huber_loss(ent_h_emb, old_ent_h_emb)
        ent_t_distill_loss = self.huber_loss(ent_t_emb, old_ent_t_emb)
        rel_distill_loss = self.huber_loss(rel_emb, old_rel_emb)
        rel_inv_distill_loss = self.huber_loss(rel_inv_emb, old_rel_inv_emb)
        
        distill_loss = ent_h_distill_loss + ent_t_distill_loss + rel_distill_loss + rel_inv_distill_loss
        self.args.distill_lambda = 0.001
        distill_loss = self.args.distill_lambda * distill_loss
        return distill_loss

    def dpo_loss(self, pos_triples, neg_triples):
        ent_h_emb, ent_t_emb, rel_emb, rel_inv_emb = self.embedding()
        old_ent_h_emb, old_ent_t_emb, old_rel_emb, old_rel_inv_emb = self.old_embeddings()
        
        pos_head, pos_relation, pos_tail = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        neg_head, neg_relation, neg_tail = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        
        # New model scores
        new_pos_h_emb = torch.index_select(ent_h_emb, 0, pos_head)
        new_pos_t_emb = torch.index_select(ent_t_emb, 0, pos_tail)
        new_pos_r_emb = torch.index_select(rel_emb, 0, pos_relation)
        new_pos_r_inv_emb = torch.index_select(rel_inv_emb, 0, pos_relation)
        
        new_neg_h_emb = torch.index_select(ent_h_emb, 0, neg_head)
        new_neg_t_emb = torch.index_select(ent_t_emb, 0, neg_tail)
        new_neg_r_emb = torch.index_select(rel_emb, 0, neg_relation)
        new_neg_r_inv_emb = torch.index_select(rel_inv_emb, 0, neg_relation)
        
        new_pos_scores = self.score_fun(pos_head, pos_relation, pos_tail, 
                                       new_pos_h_emb, new_pos_t_emb, new_pos_r_emb, new_pos_r_inv_emb)
        new_neg_scores = self.score_fun(neg_head, neg_relation, neg_tail,
                                       new_neg_h_emb, new_neg_t_emb, new_neg_r_emb, new_neg_r_inv_emb)
        
        # Old model scores
        old_pos_h_emb = torch.index_select(old_ent_h_emb, 0, pos_head)
        old_pos_t_emb = torch.index_select(old_ent_t_emb, 0, pos_tail)
        old_pos_r_emb = torch.index_select(old_rel_emb, 0, pos_relation)
        old_pos_r_inv_emb = torch.index_select(old_rel_inv_emb, 0, pos_relation)
        
        old_neg_h_emb = torch.index_select(old_ent_h_emb, 0, neg_head)
        old_neg_t_emb = torch.index_select(old_ent_t_emb, 0, neg_tail)
        old_neg_r_emb = torch.index_select(old_rel_emb, 0, neg_relation)
        old_neg_r_inv_emb = torch.index_select(old_rel_inv_emb, 0, neg_relation)
        
        old_pos_scores = self.score_fun(pos_head, pos_relation, pos_tail,
                                       old_pos_h_emb, old_pos_t_emb, old_pos_r_emb, old_pos_r_inv_emb)
        old_neg_scores = self.score_fun(neg_head, neg_relation, neg_tail,
                                       old_neg_h_emb, old_neg_t_emb, old_neg_r_emb, old_neg_r_inv_emb)
        
        # DPO loss computation
        log_softmax_new_pos = F.log_softmax(new_pos_scores, dim=-1)
        log_softmax_old_pos = F.log_softmax(old_pos_scores, dim=-1)
        log_softmax_new_neg = F.log_softmax(new_neg_scores, dim=-1)
        log_softmax_old_neg = F.log_softmax(old_neg_scores, dim=-1)
        
        self.args.dpo_beta = 1
        difference = (log_softmax_new_pos - log_softmax_old_pos) - (log_softmax_new_neg - log_softmax_old_neg)
        loss = -F.log_softmax(self.args.dpo_beta * difference, dim=-1)
        loss = torch.mean(loss)
        self.args.dpo_lamba = 0.1
        loss = self.args.dpo_lamba * loss
        return loss

    def predict(self, head, relation):
        ent_h_emb, ent_t_emb, rel_emb, rel_inv_emb = self.embedding()
        h_emb = torch.index_select(ent_h_emb, 0, head)
        r_emb = torch.index_select(rel_emb, 0, relation)
        r_inv_emb = torch.index_select(rel_inv_emb, 0, relation)
        
        # For all possible tails
        t_all_emb = ent_t_emb
        
        # Expand dimensions for broadcasting
        h_emb = h_emb.unsqueeze(1)  # [batch, 1, dim]
        r_emb = r_emb.unsqueeze(1)  # [batch, 1, dim]
        r_inv_emb = r_inv_emb.unsqueeze(1)  # [batch, 1, dim]
        t_all_emb = t_all_emb.unsqueeze(0)  # [1, num_ent, dim]
        
        # SimplEx score: <h_h, r> + <t_t, r_inv>
        score_h = torch.sum(h_emb * r_emb, dim=-1)  # [batch, 1]
        score_t = torch.sum(t_all_emb * r_inv_emb, dim=-1)  # [batch, num_ent]
        
        score = score_h + score_t
        score = torch.sigmoid(score)
        return score
    

class RotatEDPO(RotatE):
    def __init__(self, args, kg) -> None:
        super(RotatEDPO, self).__init__(args, kg)
        self.args = args
        self.kg = kg
        # RotatE uses complex embeddings for entities and phase embeddings for relations
        self.ent_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim * 2).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim).to(self.args.device).double()
        
        # Initialize entity embeddings
        xavier_normal_(self.ent_embeddings.weight)
        # Initialize relation embeddings to be within [0, 2π]
        nn.init.uniform_(self.rel_embeddings.weight, -math.pi, math.pi)
        
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None
        self.epsilon = 2.0  # Margin for distance-based scoring

    def save_embeddings(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def embedding(self):
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def old_embeddings(self):
        for name, value in self.named_buffers():
            if name == "old_data_ent_embeddings_weight":
                old_data_ent_embeddings_weight = value
            elif name == "old_data_rel_embeddings_weight":
                old_data_rel_embeddings_weight = value
        return old_data_ent_embeddings_weight, old_data_rel_embeddings_weight

    def get_complex_embeddings(self, embeddings):
        """Split embeddings into real and imaginary parts"""
        real = embeddings[:, :self.args.emb_dim]
        imag = embeddings[:, self.args.emb_dim:]
        return real, imag

    def get_rotation_matrix(self, phases):
        """Convert phase to rotation matrix components"""
        cos_phase = torch.cos(phases)
        sin_phase = torch.sin(phases)
        return cos_phase, sin_phase

    def complex_rotate(self, ent_real, ent_imag, cos_phase, sin_phase):
        """Apply rotation: (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i"""
        real = ent_real * cos_phase - ent_imag * sin_phase
        imag = ent_real * sin_phase + ent_imag * cos_phase
        return real, imag

    def complex_distance(self, h_real, h_imag, t_real, t_imag):
        """Calculate L2 distance between complex numbers"""
        real_diff = h_real - t_real
        imag_diff = h_imag - t_imag
        return torch.sqrt(real_diff**2 + imag_diff**2)

    def score_fun(self, h, r, t):
        h_real, h_imag = self.get_complex_embeddings(h)
        t_real, t_imag = self.get_complex_embeddings(t)
        cos_phase, sin_phase = self.get_rotation_matrix(r)
        
        # Rotate head: h * exp(i*r)
        h_rot_real, h_rot_imag = self.complex_rotate(h_real, h_imag, cos_phase, sin_phase)
        
        # Calculate distance: ||h ○ r - t||
        distance = self.complex_distance(h_rot_real, h_rot_imag, t_real, t_imag)
        return torch.sum(distance, dim=-1)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def margin_loss(self, head, relation, tail, label=None):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.margin_loss(head, relation, tail, label) / head.size(0)
        return loss

    def distill_loss(self):
        ent_embeddings, rel_embeddings = self.embedding()
        self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)
        ent_embeddings = ent_embeddings * self.entity_distill_mask.unsqueeze(1)
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()
        old_data_ent_embeddings = old_data_ent_embeddings * self.entity_distill_mask.unsqueeze(1)
        ent_distill_loss = self.huber_loss(ent_embeddings, old_data_ent_embeddings)
        rel_distill_loss = self.huber_loss(rel_embeddings, old_data_rel_embeddings)
        distill_loss = ent_distill_loss + rel_distill_loss
        self.args.distill_lambda = 0.001
        distill_loss = self.args.distill_lambda * distill_loss
        return distill_loss

    def dpo_loss(self, pos_triples, neg_triples):
        ent_embeddings, rel_embeddings = self.embedding()
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()
        pos_head, pos_relation, pos_tail = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        neg_head, neg_relation, neg_tail = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        
        # New model scores
        new_pos_h = torch.index_select(ent_embeddings, 0, pos_head)
        new_pos_r = torch.index_select(rel_embeddings, 0, pos_relation)
        new_pos_t = torch.index_select(ent_embeddings, 0, pos_tail)
        new_neg_h = torch.index_select(ent_embeddings, 0, neg_head)
        new_neg_r = torch.index_select(rel_embeddings, 0, neg_relation)
        new_neg_t = torch.index_select(ent_embeddings, 0, neg_tail)
        new_pos_scores = self.score_fun(new_pos_h, new_pos_r, new_pos_t)
        new_neg_scores = self.score_fun(new_neg_h, new_neg_r, new_neg_t)
        
        # Old model scores
        old_pos_h = torch.index_select(old_data_ent_embeddings, 0, pos_head)
        old_pos_r = torch.index_select(old_data_rel_embeddings, 0, pos_relation)
        old_pos_t = torch.index_select(old_data_ent_embeddings, 0, pos_tail)
        old_neg_h = torch.index_select(old_data_ent_embeddings, 0, neg_head)
        old_neg_r = torch.index_select(old_data_rel_embeddings, 0, neg_relation)
        old_neg_t = torch.index_select(old_data_ent_embeddings, 0, neg_tail)
        old_pos_scores = self.score_fun(old_pos_h, old_pos_r, old_pos_t)
        old_neg_scores = self.score_fun(old_neg_h, old_neg_r, old_neg_t)
        
        # DPO loss computation
        log_softmax_new_pos = F.log_softmax(new_pos_scores, dim=-1)
        log_softmax_old_pos = F.log_softmax(old_pos_scores, dim=-1)
        log_softmax_new_neg = F.log_softmax(new_neg_scores, dim=-1)
        log_softmax_old_neg = F.log_softmax(old_neg_scores, dim=-1)
        
        self.args.dpo_beta = 1
        difference = (log_softmax_new_pos - log_softmax_old_pos) - (log_softmax_new_neg - log_softmax_old_neg)
        loss = -F.log_softmax(self.args.dpo_beta * difference, dim=-1)
        loss = torch.mean(loss)
        self.args.dpo_lamba = 0.1
        loss = self.args.dpo_lamba * loss
        return loss

    def predict(self, head, relation):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t_all = ent_embeddings
        
        h_real, h_imag = self.get_complex_embeddings(h)
        t_all_real, t_all_imag = self.get_complex_embeddings(t_all)
        cos_phase, sin_phase = self.get_rotation_matrix(r)
        
        # Rotate head: h * exp(i*r)
        h_rot_real, h_rot_imag = self.complex_rotate(h_real, h_imag, cos_phase, sin_phase)
        
        # Expand for all entities
        h_rot_real = h_rot_real.unsqueeze(1)
        h_rot_imag = h_rot_imag.unsqueeze(1)
        t_all_real = t_all_real.unsqueeze(0)
        t_all_imag = t_all_imag.unsqueeze(0)
        
        # Calculate distance: ||h ○ r - t||
        distance = self.complex_distance(h_rot_real, h_rot_imag, t_all_real, t_all_imag)
        distance = torch.sum(distance, dim=-1)
        
        # Convert distance to similarity score
        score = self.epsilon - distance
        score = torch.sigmoid(score)
        return score
from ..utils import *
from src.model.TransE import TransE

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
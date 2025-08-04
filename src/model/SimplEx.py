from ..utils import *

class SimplEx(nn.Module):
    def __init__(self, args, kg) -> None:
        super(SimplEx, self).__init__()
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

    def embedding(self):
        return (self.ent_h_embeddings.weight, self.ent_t_embeddings.weight, 
                self.rel_embeddings.weight, self.rel_inv_embeddings.weight)

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
    
    def forward(self, head, relation=None):
        assert self.args.unlearning_method == "ADVIMP"
        n = head.size(0)
        half_n = n // 2
        h, r = head[:half_n], head[half_n:]
        
        ent_h_emb, ent_t_emb, rel_emb, rel_inv_emb = self.embedding()
        h_emb = torch.index_select(ent_h_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)
        r_inv_emb = torch.index_select(rel_inv_emb, 0, r)
        
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

class UnSimplEx(SimplEx):
    def __init__(self, args, kg) -> None:
        super(UnSimplEx, self).__init__(args, kg)
        self.args = args
        self.kg = kg
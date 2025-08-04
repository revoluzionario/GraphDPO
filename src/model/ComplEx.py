from ..utils import *

class ComplEx(nn.Module):
    def __init__(self, args, kg) -> None:
        super(ComplEx, self).__init__()
        self.args = args
        self.kg = kg
        # ComplEx uses complex embeddings, so we need 2x dimensions
        self.ent_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim * 2).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim * 2).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")

    def embedding(self):
        return self.ent_embeddings.weight, self.rel_embeddings.weight

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
    
    def forward(self, head, relation=None):
        assert self.args.unlearning_method == "ADVIMP"
        n = head.size(0)
        half_n = n // 2
        h, r = head[:half_n], head[half_n:]
        
        ent_embeddings, rel_embeddings = self.embedding()
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

class UnComplEx(ComplEx):
    def __init__(self, args, kg) -> None:
        super(UnComplEx, self).__init__(args, kg)
        self.args = args
        self.kg = kg
from ..utils import *
import math

class RotatE(nn.Module):
    def __init__(self, args, kg) -> None:
        super(RotatE, self).__init__()
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
        self.epsilon = 2.0  # Margin for distance-based scoring

    def embedding(self):
        return self.ent_embeddings.weight, self.rel_embeddings.weight

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
    
    def forward(self, head, relation=None):
        assert self.args.unlearning_method == "ADVIMP"
        n = head.size(0)
        half_n = n // 2
        h, r = head[:half_n], head[half_n:]
        
        ent_embeddings, rel_embeddings = self.embedding()
        h_emb = torch.index_select(ent_embeddings, 0, h)
        r_emb = torch.index_select(rel_embeddings, 0, r)
        t_all = ent_embeddings
        
        h_real, h_imag = self.get_complex_embeddings(h_emb)
        t_all_real, t_all_imag = self.get_complex_embeddings(t_all)
        cos_phase, sin_phase = self.get_rotation_matrix(r_emb)
        
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

class UnRotatE(RotatE):
    def __init__(self, args, kg) -> None:
        super(UnRotatE, self).__init__(args, kg)
        self.args = args
        self.kg = kg
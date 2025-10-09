import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
import sys
sys.path.append("/train20/.../ECAPA-TDNN-main")
from ECAPAModel import ECAPAModel

def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def sisnr(x, s, eps=1e-8):
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def compute_sisnr_loss(ests, refs):
    num_spks = len(refs)
    N = refs[0].size(0)

    def sisnr_loss(permute):
        return sum([sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]) / num_spks

    sisnr_mat = torch.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, best_perms = torch.max(sisnr_mat, dim=0)
    sisnr_loss_value = -torch.sum(max_perutt) / N
    return sisnr_loss_value, best_perms

class SpeakerSimilarityLoss(nn.Module):
    def __init__(self, model_path="/train20/.../ECAPA-TDNN-main/exps/pretrain.model"):
        super().__init__()
        # 加载预训练的ECAPA-TDNN模型
        self.timbre_model = ECAPAModel()
        self.timbre_model.load_parameters(model_path)
        self.timbre_model.eval()
        # 冻结模型参数
        for param in self.timbre_model.parameters():
            param.requires_grad = False

    def extract_embedding(self, audio_tensor):
        with torch.no_grad():
            # 提取嵌入并归一化
            embeddings = self.timbre_model.speaker_encoder(audio_tensor.squeeze(1), aug=False)
            return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, ests, refs, best_perms):
        B = ests[0].size(0)  # 批量大小
        num_spks = 2
        # 提取所有嵌入
        est_embeddings = [self.extract_embedding(est) for est in ests]  # [est1_emb, est2_emb]
        ref_embeddings = [self.extract_embedding(ref) for ref in refs]  # [ref1_emb, ref2_emb]
        # 初始化损失
        same_spk_sim = 0
        diff_spk_sim = 0
        # 根据最佳排列计算相似性
        for b in range(B):
            # 获取当前样本的最佳排列
            perm_idx = best_perms[b].item()
            if perm_idx == 0:
                permute = (0, 1)
            else:
                permute = (1, 0)
            # 计算同源相似性（s_i与est_i）
            for i in range(num_spks):
                ref_emb = ref_embeddings[i][b]  # 第 i 个参考源的嵌入
                est_emb = est_embeddings[permute[i]][b]  # 第 i 个参考源对应的最佳估计源
                same_spk_sim += torch.dot(ref_emb, est_emb)
            # 计算异源相似性（est_0与est_1）
            est1_emb = est_embeddings[0][b]
            est2_emb = est_embeddings[1][b]
            diff_spk_sim += torch.dot(est1_emb, est2_emb)
        # 构造损失函数
        L_spk = -same_spk_sim / B + diff_spk_sim / B
        return L_spk
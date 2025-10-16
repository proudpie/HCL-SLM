import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from transformers import WavLMModel, AutoProcessor, Wav2Vec2FeatureExtractor
class WavLMLoss(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-large"):
        super().__init__()
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def extract_semantic_features(self, audio_signal):
        """从音频信号中提取语义特征"""
        input_features = audio_signal # 不需要过处理器
        encoder_outputs = self.model(input_features)
        hidden_states = encoder_outputs.last_hidden_state # [batch, seq_len, dim]
        features = F.layer_norm(hidden_states, (hidden_states.size(-1),))

        return F.normalize(features, p=2, dim=-1)
    
    def sim(self, generated_audio, reference_audio):
        generated_audio = generated_audio - torch.mean(generated_audio)
        generated_audio = generated_audio / torch.max(torch.abs(generated_audio))
        gen_features = self.extract_semantic_features(generated_audio)
        ref_features = self.extract_semantic_features(reference_audio)
        cos_sim = F.cosine_similarity(gen_features, ref_features, dim=-1)  # [B, T]
        cos_sim = cos_sim.mean(dim=-1)  # [B]
        sim_loss = 1 - cos_sim
        return sim_loss  # [B]

    def forward(self, ests, refs):
        num_spks = len(refs)
        def compute_sim_score(permute):
            return sum([self.sim(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        loss_mat = torch.stack([compute_sim_score(p) for p in permutations(range(num_spks))])
        min_perutt, _ = torch.min(loss_mat, dim=0)

        return min_perutt.mean()

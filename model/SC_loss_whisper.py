import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor_2
from itertools import permutations

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

class SC_Loss(nn.Module):
    def __init__(self, model_name="/dmx-csy-mix01/.../whisper_small"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).model.encoder
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.length = 200  # 有效长度

    def _audio_to_features(self, audio_signal):
        inputs, _ = self.processor(audio_signal, sampling_rate=16000, return_tensors='pt')
        return inputs

    def _compute_energy_mask(self, audio_signal):
        B, T = audio_signal.shape
        num_blocks = T // 320
        audio_signal = audio_signal[:, :num_blocks * 320].view(B, num_blocks, 320)
        energy = (audio_signal ** 2).sum(dim=-1)
        return (energy > 0.5).float()

    def extract_semantic_features(self, audio_signal):
        input_features = self._audio_to_features(audio_signal)
        encoder_outputs = self.model(input_features.squeeze(1))
        hidden_states = encoder_outputs.last_hidden_state
        hidden_states = hidden_states[:, :self.length, :]
        return F.normalize(F.layer_norm(hidden_states, (hidden_states.size(-1),)), p=2, dim=-1)
    
    def sim(self, generated_audio, reference_audio, mask, threshold=0.7):
        generated_audio = generated_audio - torch.mean(generated_audio)
        generated_audio = generated_audio / torch.max(torch.abs(generated_audio))
        gen_features = self.extract_semantic_features(generated_audio)
        ref_features = self.extract_semantic_features(reference_audio)
        cos_sim = F.cosine_similarity(gen_features, ref_features, dim=-1)
        thresholded_loss = torch.clamp(cos_sim - threshold, min=0) * mask * 10
        sim_loss = 1 - cos_sim.mean(dim=-1)
        sc_loss = thresholded_loss.mean(dim=-1)
        return sim_loss, sc_loss

    def compute_sim_score(self, permute, mask, ests, refs):
        losses_sim = []
        losses_sc = []
        for s, t in enumerate(permute):
            sim_loss, sc_loss = self.sim(ests[s], refs[t], mask)
            losses_sim.append(sim_loss)
            losses_sc.append(sc_loss)
        avg_sim_loss = torch.stack(losses_sim).mean(dim=0)
        avg_sc_loss = torch.stack(losses_sc).mean(dim=0)
        return avg_sim_loss, avg_sc_loss
    
    def forward(self, ests, refs, best_perm_indices):
        num_spks = len(refs)
        batch_size = ests[0].size(0)
        device = ests[0].device
        mask0 = self._compute_energy_mask(ests[0]).to(device)
        mask1 = self._compute_energy_mask(ests[1]).to(device)
        mask = torch.max(mask0, mask1)
        permutations_list = list(permutations(range(num_spks)))
        loss_mat_sim = torch.zeros(len(permutations_list), batch_size, device=device)
        loss_mat_sc = torch.zeros(len(permutations_list), batch_size, device=device)
        for p_idx, permute in enumerate(permutations_list):
            sim_loss, sc_loss = self.compute_sim_score(permute, mask, ests, refs)
            loss_mat_sim[p_idx] = sim_loss
            loss_mat_sc[p_idx] = sc_loss

        best_sim_loss = loss_mat_sim[best_perm_indices, torch.arange(batch_size, device=device)].mean()

        cross_loss_total = 0.0
        for b in range(batch_size):
            best_idx = best_perm_indices[b]
            for p_idx in range(len(permutations_list)):
                if p_idx != best_idx:
                    cross_loss_total += loss_mat_sc[p_idx, b]

        cross_loss_total /= ((len(permutations_list) - 1) * batch_size)
        return best_sim_loss, cross_loss_total

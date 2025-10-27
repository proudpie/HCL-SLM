import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
from transformers import AutoTokenizer, WhisperFeatureExtractor_2, Qwen2ForCausalLM
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
    def __init__(self, model_name="/dmx-csy-mix01/.../qwen2audio"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # 只用编码器的特征层，解码器需要输出prompt
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name).audio_tower
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.length = 100
        # self.prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        self.prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Analyse and conclude what you hear:"

    def _audio_to_features(self, audio_signal):
        inputs = self.processor(
            text = self.prompt,
            audios = audio_signal,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs["input_ids"] = inputs["input_ids"].to(audio_signal.device)
        return inputs

    def _compute_energy_mask(self, audio_signal):
        B, T = audio_signal.shape
        num_blocks = T // 640
        audio_signal = audio_signal[:, :num_blocks * 640].view(B, num_blocks, 640)
        energy = (audio_signal ** 2).sum(dim=-1)
        return (energy > 0.5).float()

    def extract_semantic_features(self, audio_signal):
        inputs = self._audio_to_features(audio_signal)
        hidden_states = self.extract(**inputs)
        hidden_states = hidden_states[:, :self.length, :]
        # print("hidden_states:", hidden_states.shape)
        return F.normalize(F.layer_norm(hidden_states, (hidden_states.size(-1),)), p=2, dim=-1)
    
    def sim(self, generated_audio, reference_audio, mask, threshold=0.7):
        generated_audio = generated_audio - torch.mean(generated_audio)
        generated_audio = generated_audio / torch.max(torch.abs(generated_audio))
        gen_features = self.extract_semantic_features(generated_audio)
        ref_features = self.extract_semantic_features(reference_audio)
        cos_sim = F.cosine_similarity(gen_features, ref_features, dim=-1)
        thresholded_loss = torch.clamp(cos_sim - threshold, min=0) * mask * 5
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

    def extract(
        self,
        input_ids,
        input_features,
        attention_mask,
        feature_attention_mask,
    ):
        target_device = self.model.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        if input_features is not None and input_ids.shape[1] != 1:
            audio_feat_lengths, audio_output_lengths = self.model._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.model.conv1.weight.dtype, device=self.model.conv1.weight.device
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_outputs = self.model(input_features, attention_mask=audio_attention_mask)
            selected_audio_feature = audio_outputs.last_hidden_state
        return selected_audio_feature

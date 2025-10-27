import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
from transformers import AutoTokenizer, WhisperFeatureExtractor_2, Qwen2ForCausalLM
from itertools import permutations

class QwenLoss(nn.Module):
    def __init__(self, model_name="/dmx-csy-mix01/.../qwen2audio"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # 只用编码器的特征层，解码器需要输出prompt
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.length = 200 # whisper处理30秒输出1500维，训练数据4秒200维有效
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
    
    def extract_semantic_features(self, audio_signal):
        """从音频信号中提取语义特征"""
        inputs = self._audio_to_features(audio_signal)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        features = F.layer_norm(hidden_states, (hidden_states.size(-1),))
        return F.normalize(features, p=2, dim=-1)

    def sim(self, generated_audio, reference_audio):
        """计算生成音频和参考音频的语义相似度损失"""
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
            # for one permute, average the value
            return sum([self.sim(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        loss_mat = torch.stack([compute_sim_score(p) for p in permutations(range(num_spks))]) # 得到 [(0,1), (1,0)]
        min_perutt, _ = torch.min(loss_mat, dim=0) # 沿排列维度取最大值

        return min_perutt.mean()


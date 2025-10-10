# Hierarchical Contrastive Learning with Speech Language Model for Separating Similar Speakers (HCL-SLM)

Current learning objectives for blind source separation (BSS) mainly focus on reducing inter-speaker differences. However, in real-world scenarios where mixed speakers exhibit high timbre similarity, state-of-the-art (SOTA) models degrade severely. To evaluate and address this challenge, we first construct the WSJ0-same-mix benchmark, which mixes different utterances from the same speaker to simulate extreme similarity conditions. We further propose a Hierarchical Contrastive Learning with Speech Language Model (HCL-SLM) framework, which introduces dual contrastive losses to exploit layer-specific representations from Qwen2-Audio, pulling together segments from the same speaker while pushing apart those from different speakers. Extensive experiments demonstrate that several representative SOTA BSS models suffer significant degradation on the WSJ0-same-mix benchmark, whereas HCL-SLM consistently improves their performance.

## Pipeline
<img width="2336" height="862" alt="pipeline" src="https://github.com/user-attachments/assets/9e272660-94e1-4d1a-8ca3-c0963ff7c7c9" />

## Experimental Results
<img width="1376" height="346" alt="result_table1" src="https://github.com/user-attachments/assets/0de058f5-5cb6-4db3-9b0b-083368cef823" />

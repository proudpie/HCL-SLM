import os
import random
from tqdm import tqdm

# 配置
DATASETS = ["/train20/.../datasets/WSJ0-2mix-/wsj0/si_et_05",
            "/train20/.../datasets/WSJ0-2mix-/wsj0/si_dt_05"]  # 两个数据集目录
OUTPUT_FILE = "/train20/.../mypipeline/process/metadata/mix_2_spk_tt_xx.txt"
NUM_MIX = 1000

def get_valid_wav_files(spk_dir):
    """获取该说话人目录下所有不以 _1.wav 结尾的 wav 文件"""
    return [f for f in os.listdir(spk_dir) if f.endswith('.wav') and not f.endswith('_1.wav')]

def collect_speakers(dataset_paths):
    """从所有数据集中收集说话人目录"""
    all_speakers = []
    for root in dataset_paths:
        if not os.path.exists(root):
            print(f"警告: 目录不存在, 跳过: {root}")
            continue
        spk_ids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for spk in spk_ids:
            spk_path = os.path.join(root, spk)
            all_speakers.append(spk_path)
    return all_speakers

def main():
    print("开始收集说话人目录 ...")
    spk_dirs = collect_speakers(DATASETS)
    print(f"共找到 {len(spk_dirs)} 个说话人目录")

    if not spk_dirs:
        raise ValueError("未找到任何有效的说话人目录，请检查路径是否正确")
    
    with open(OUTPUT_FILE, 'w') as f:
        pbar = tqdm(total=NUM_MIX, desc="生成混合语音对")
        generated = 0

        while generated < NUM_MIX:
            # 随机选择一个说话人目录
            spk_dir = random.choice(spk_dirs)
            spk_id = os.path.basename(spk_dir)
            dataset_prefix = os.path.basename(os.path.dirname(spk_dir))  # 获取是 si_tr_s 还是 si_et_05

            wav_files = get_valid_wav_files(spk_dir)
            if len(wav_files) < 2:
                continue  # 语音不足两个，跳过

            # 随机选择两个不同语音
            wav1, wav2 = random.sample(wav_files, 2)

            # SNR 设置
            snr1 = round(random.uniform(-1, 1), 1)
            snr2 = -snr1

            # 构造语音路径（相对于 wsj0/）
            path1 = f"wsj0/{dataset_prefix}/{spk_id}/{wav1}"
            path2 = f"wsj0/{dataset_prefix}/{spk_id}/{wav2}"

            # 写入文件
            line = f'{path1} {snr1} {path2} {snr2}\n'
            f.write(line)
            generated += 1
            pbar.update(1)

        pbar.close()

    print(f"混合语音对已生成至；{OUTPUT_FILE}")

if __name__ == "__main__":
    main()

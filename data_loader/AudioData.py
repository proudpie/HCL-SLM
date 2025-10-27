import torch.nn.functional as F
from utils import util
import torch
import torchaudio
import sys
sys.path.append('../')
import time
import tqdm


def read_wav(fname, return_rate=False):
    '''
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L 
                     L is the number of audio frames 
                     C is the number of channels. 
                sr: sample rate
    '''
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()


def write_wav(fname, src, sample_rate):
    '''
         Write wav file
         input:
               fname: wav file path
               src: frames of audio
               sample_rate: An integer which is the sample rate of the audio
         output:
               None
    '''
    torchaudio.save(fname, src, sample_rate)

# 添加进度条装饰器
def progress_bar(func):
    def wrapper(*args, **kwargs):
        print(f"开始 {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} 完成! 耗时: {elapsed:.2f}秒")
        return result
    return wrapper

class AudioReader(object):
    '''
        Class that reads Wav format files
        Input:
            scp_path (str): a different scp file address
            sample_rate (int, optional): sample rate (default: 8000)
            chunk_size (int, optional): split audio size (default: 32000(4 s))
            least_size (int, optional): Minimum split size (default: 16000(2 s))
        Output:
            split audio (list)
        读取音频文件并分割成固定长度的片段
        短于16000丢弃
        介于最小长度和片段长度之间：在末尾填充0值（静音）
        长于≥32000点使用滑动窗口切分（窗口大小=32000，步长=16000叠）
    '''

    def __init__(self, scp_path, sample_rate=8000, chunk_size=32000, least_size=16000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = util.handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())
        self.audio = []
        self.chunk_size = chunk_size
        self.least_size = least_size
        self.split()

    def split(self):
        '''
            split audio with chunk_size and least_size
        '''
        total_files = len(self.keys)
        valid_files = 0
        total_segments = 0
        
        # 使用tqdm显示进度条
        iterator = tqdm.tqdm(self.keys, desc="处理音频文件")

        for key in iterator:
            utt = read_wav(self.index_dict[key])
            if utt.shape[0] < self.least_size:
                continue
            valid_files += 1
            if utt.shape[0] > self.least_size and utt.shape[0] < self.chunk_size:
                gap = self.chunk_size-utt.shape[0]
                self.audio.append(F.pad(utt, (0, gap), mode='constant'))
                total_segments += 1
            if utt.shape[0] >= self.chunk_size:
                start = 0
                while True:
                    if start + self.chunk_size > utt.shape[0]:
                        break
                    self.audio.append(utt[start:start+self.chunk_size])
                    start += self.least_size
                    total_segments += 1
                # 更新进度条描述
                    iterator.set_description(f"处理音频文件 (已处理 {valid_files}/{total_files} 文件, {total_segments} 片段)")
        # 最终统计信息
        print(f"\n音频处理完成!")
        print(f"总文件数: {total_files}")
        print(f"有效文件数: {valid_files} ({valid_files/total_files*100:.1f}%)")
        print(f"生成片段数: {total_segments}")
        print(f"平均每个有效文件: {total_segments/max(1, valid_files):.1f} 片段")


if __name__ == "__main__":
    a = AudioReader("/home/likai/data1/create_scp/cv_mix.scp")
    audio = a.audio
    print(len(audio))

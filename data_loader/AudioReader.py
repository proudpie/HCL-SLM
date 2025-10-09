import sys
sys.path.append('../')
import torchaudio
import torch
from utils.util import handle_scp


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


class AudioReader(object):
    '''
        Class that reads Wav format files
        Input as a different scp file address
        Output a matrix of wav files in all scp files.
        读取完整长度音频文件
    '''

    def __init__(self, scp_path, sample_rate=8000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())

    def _load(self, key):
        src, sr = read_wav(self.index_dict[key], return_rate=True)
        if self.sample_rate is not None and sr != self.sample_rate:
            raise RuntimeError('SampleRate mismatch: {:d} vs {:d}'.format(
                sr, self.sample_rate))
        return src

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for key in self.keys:
            yield key, self._load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError('Unsupported index type: {}'.format(type(index)))
        if type(index) == int:
            num_uttrs = len(self.keys)
            if num_uttrs < index and index < 0:
                raise KeyError('Interger index out of range, {:d} vs {:d}'.format(
                    index, num_uttrs))
            index = self.keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))

        return self._load(index)
    
# class AudioReader2(object): # future work
#     def __init__(self, scp_path1, scp_path2, sample_rate=8000):
#         super(AudioReader, self).__init__()
#         self.sample_rate = sample_rate

#         # 读取两个scp文件
#         self.index_dict1 = handle_scp(scp_path1)
#         self.index_dict2 = handle_scp(scp_path2)

#         # 验证两个scp文件的utterance ID是否一致
#         if set(self.index_dict1.keys()) != set(self.index_dict2.keys()):
#             raise ValueError("Keys in the two .scp files do not match.")

#         self.keys = list(self.index_dict1.keys())

#     def _load(self, key):
#         # 从两个scp中分别读取对应路径的音频
#         src1, sr1 = read_wav(self.index_dict1[key], return_rate=True)
#         src2, sr2 = read_wav(self.index_dict2[key], return_rate=True)
#         return src1, src2
    
#     def __len__(self):
#         return len(self.keys)

#     def __iter__(self):
#         for key in self.keys:
#             yield key, self._load(key)

#     def __getitem__(self, index):
#         if not isinstance(index, (int, str)):
#             raise IndexError(f'Unsupported index type: {type(index)}')
#         if isinstance(index, int):
#             num_uttrs = len(self.keys)
#             if index < 0 or index >= num_uttrs:
#                 raise KeyError(f'Integer index out of range: {index} vs {num_uttrs}')
#             index = self.keys[index]
#         if index not in self.index_dict1 or index not in self.index_dict2:
#             raise KeyError(f"Missing utterance {index} in one or both .scp files")
#         return self._load(index)

if __name__ == "__main__":
    r = AudioReader('/home/likai/data1/create_scp/cv_s2.scp')
    index = 0
    print(r[1])

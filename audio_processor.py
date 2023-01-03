"""
Process wave audio file into format feature

===

Depends:
    numpy
    scipy
    ffmpeg
    imageio
    librosa = 0.9.2
"""
import os
import torch
import audio_utils
import numpy as np
from functools import partial


class APCProcessor:
    def __init__(self, ckpt_fp='data/APC_epoch_160.model', is_gpu=True) -> None:
        self.is_gpu = is_gpu
        self.processor = audio_utils.APC_encoder(
                                80,     # self.opt.audiofeature_input_channels,
                                512,    # self.opt.APC_hidden_size,
                                3,      # self.opt.APC_rnn_layers,
                                False,  # self.opt.APC_residual
                                )
        self.processor.load_state_dict(torch.load(ckpt_fp), strict=False)
        if is_gpu: self.processor.cuda()
        self.processor.eval()
    
    def __call__(self, wav):
        mel80 = audio_utils.melspectrogram(wav)
        mel_nframe = mel80.shape[0]
        length = torch.Tensor([mel_nframe])
        mel80_torch = torch.from_numpy(mel80.astype(np.float32)).unsqueeze(0)
        if self.is_gpu: mel80_torch = mel80_torch.cuda()
        hidden_reps = self.processor.forward(mel80_torch, length)[0]   # [mel_nframe, 512]
        hidden_reps = hidden_reps.cpu().numpy()

        return hidden_reps


class AudioProcessor(object):
    def __init__(self, proc_type='mel', save_dir=None, sample_rate=16000, fps=30) -> None:
        """Audio processor

        Args:
            proc_type (str, optional): Choices from ['mel', 'wav2vec', 'apc', 'none']. Defaults to 'mel'.
            save_dir (str, optional): Saved audio feature dir, if not none, excute check_read and save. Defaults to None.
            sample_rate (int, optional): Loading wav sample rate. Defaults to 16000.
            fps (int, optional): Target video fps. Defaults to 30.
        """
        self.proc_type = proc_type.lower()
        self.save_dir = save_dir
        self.sample_rate = sample_rate
        self.fps = fps

        self.wav_reader = partial(audio_utils.load_wav, sr=self.sample_rate)

        if self.proc_type == 'mel':
            self.processor = audio_utils.melspectrogram
        elif self.proc_type == 'apc':
            self.processor = APCProcessor()
        elif self.proc_type == 'wav2vec':
            from transformers import Wav2Vec2Processor
            wvp = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.processor = lambda x: np.squeeze(wvp(x, sampling_rate=self.sample_rate).input_values)
        elif self.proc_type == 'none':
            self.processor = lambda x: x
        else:
            raise NotImplementedError(f'Unsupported `proc_type`: {self.proc_type}')

    def process(self, wav, fn=None):
        if type(wav) is str:
            fn = os.path.basename(wav).split('.')[0]
            _wav = self.wav_reader(wav)
        else:
            fn = os.path.basename(fn).split('.')[0]
            _wav = wav
        
        if self.proc_type != 'none':
            if self.save_dir is not None:
                # check & read
                feat_fp = os.path.join(self.save_dir, f'{fn}-{self.proc_type}.npy')
                if os.path.exists(feat_fp): 
                    return np.load(feat_fp)
                else:
                    os.makedirs(self.save_dir, exist_ok=True)
                    ret = self.processor(_wav)
                    np.save(feat_fp, ret)
                    return ret
        
        return self.processor(_wav)

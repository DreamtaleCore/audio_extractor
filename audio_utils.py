import torch
import librosa
import numpy as np
import torch.nn as nn
import librosa.filters
from scipy import signal
from scipy.io import wavfile
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis_fn(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis_fn(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def linearspectrogram(wav, signal_normalization=True, preemphasize=True, preemphasis=0.97, ref_level_db=20,):
    D = _stft(preemphasis_fn(wav, preemphasis, preemphasize))
    S = _amp_to_db(np.abs(D)) - ref_level_db
    
    if signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav, signal_normalization=True, preemphasize=True, preemphasis=0.97, ref_level_db=20):
    D = _stft(preemphasis_fn(wav, preemphasis, preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db
    
    if signal_normalization:
        return _normalize(S)
    return S


def _stft(y, 
          n_fft=800,        # Extra window size is filled with 0 paddings to match this parameter
	      hop_size=200,     # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
	      win_size=800):    # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis(
	                 fmin=55, fmax=7600,    # To be increased/reduced depending on data
                     sample_rate=16000,     # 16000Hz (corresponding to librispeech) (sox --i <filename>)
                     n_fft=800,             # Extra window size is filled with 0 paddings to match this parameter
                     num_mels=80,           # Number of mel-spectrogram channels and local conditioning dimensionality
                    ):
    assert fmax <= sample_rate // 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels,
                               fmin=fmin, fmax=fmax)

def _amp_to_db(x, min_level_db=-100):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
	              symmetric_mels=True,                   # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence
                  max_abs_value=4.,                      # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)
	              min_level_db=-100):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                           -max_abs_value, max_abs_value)
        else:
            return np.clip(max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value)
    
    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))

def _denormalize(D, allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
	                symmetric_mels=True,                   # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence
                    max_abs_value=4.,                      # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)
	                min_level_db=-100):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return (((np.clip(D, -max_abs_value,
                              max_abs_value) + max_abs_value) * -min_level_db / (2 * max_abs_value))
                    + min_level_db)
        else:
            return ((np.clip(D, 0, max_abs_value) * -min_level_db / max_abs_value) + min_level_db)
    
    if symmetric_mels:
        return (((D + max_abs_value) * -min_level_db / (2 * max_abs_value)) + min_level_db)
    else:
        return ((D * -min_level_db / max_abs_value) + min_level_db)



class APC_encoder(nn.Module):
    def __init__(self,
                 mel_dim,
                 hidden_size,
                 num_layers,
                 residual):
        super(APC_encoder, self).__init__()

        input_size = mel_dim

        in_sizes = ([input_size] + [hidden_size] * (num_layers - 1))
        out_sizes = [hidden_size] * num_layers
        self.rnns = nn.ModuleList(
                [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_residual = residual
    
    def forward(self, inputs, lengths):
        '''
        input:
            inputs: (batch_size, seq_len, mel_dim)
            lengths: (batch_size,)

        return:
            predicted_mel: (batch_size, seq_len, mel_dim)
            internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
            where x is 1 if there's a prenet, otherwise 0
        '''
        with torch.no_grad():
            seq_len = inputs.size(1)
            packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)
        
            for i, layer in enumerate(self.rnns):
                packed_rnn_outputs, _ = layer(packed_rnn_inputs)
                
                rnn_outputs, _ = pad_packed_sequence(
                        packed_rnn_outputs, True, total_length=seq_len)
                # outputs: (batch_size, seq_len, rnn_hidden_size)
                
                if i + 1 < len(self.rnns):
                    rnn_inputs, _ = pad_packed_sequence(
                            packed_rnn_inputs, True, total_length=seq_len)
                    # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)
                    if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                        # Residual connections
                        rnn_outputs = rnn_outputs + rnn_inputs
                    packed_rnn_inputs = pack_padded_sequence(rnn_outputs, lengths, True)
        
        
        return rnn_outputs
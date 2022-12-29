"""
Step-0: Extract audio file from video

===

Extract audio to wav file from the video file.
Depends:
    numpy
    scipy
    ffmpeg
    imageio
    librosa = 0.9.2
"""

__author__ = 'dreamtale'

import os
import imageio
import argparse
import audio_utils
import numpy as np


def extract_audio_from_video(in_v_fp, out_wav_fp, out_mel_fp=None, out_mel_chunk=True):
    
    if os.path.isdir(out_wav_fp):
        out_wav_fp = os.path.join(out_wav_fp, os.path.basename(in_v_fp).spilt('.')[0] + '-audio.wav')
    
    os.makedirs(os.path.dirname(out_wav_fp), exist_ok=True)
    extract_wav_cmd = f'ffmpeg -y -i {in_v_fp} -strict -2 {out_wav_fp}'
    os.system(extract_wav_cmd)

    wav = audio_utils.load_wav(args.out_audio_fp, 16000)

    ret_dict = {'wav': wav}

    if out_mel_fp is not None:
        mel = audio_utils.melspectrogram(wav)
        os.makedirs(os.path.dirname(out_mel_fp), exist_ok=True)
        np.save(out_mel_fp, mel)
        ret_dict['mel'] = mel
    
        if out_mel_chunk:
            reader = imageio.get_reader(in_v_fp)
            fps    = reader.get_meta_data()['fps']
            mel_step_size = 16
            mel_chunks = []         # Here `mel_chunks` store mel spec chuncks for each frame
            mel_idx_multiplier = 80./fps 
            i = 0
            while True:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + mel_step_size > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
                i += 1
            
            ret_dict['mel_chunks'] = mel_chunks
    
    return ret_dict
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio info')
    parser.add_argument('-i',  '--in_video_fp',   type=str,  help='Input video file path.')
    parser.add_argument('-o1', '--out_audio_fp',  type=str,  help='Output audio wave file path.')
    parser.add_argument('-o2', '--out_mel_fp',    type=str,  help='Output audio mel file path.')
    parser.add_argument('-o3', '--out_mel_chunk', action='store_true', help='Whether to output audio mel chuncks, which is divided by video fps')

    args = parser.parse_args()

    ret = extract_audio_from_video(args.in_video_fp, args.out_audio_fp, args.out_mel_fp)

    print(ret['wav'].shape)
    print(ret['mel'].shape)
    print(len(ret['mel_chunks']))



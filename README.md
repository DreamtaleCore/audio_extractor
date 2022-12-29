# Extract audio info from video


## Quickstart

```shell
pip install -r requirements.txt
```

Usage:

```shell
usage: extract_audio.py [-h] [-i IN_VIDEO_FP] [-o1 OUT_AUDIO_FP] [-o2 OUT_MEL_FP] [-o3]

Extract audio info

optional arguments:
  -h, --help            show this help message and exit
  -i IN_VIDEO_FP, --in_video_fp IN_VIDEO_FP
                        Input video file path.
  -o1 OUT_AUDIO_FP, --out_audio_fp OUT_AUDIO_FP
                        Output audio wave file path.
  -o2 OUT_MEL_FP, --out_mel_fp OUT_MEL_FP
                        Output audio mel file path.
  -o3, --out_mel_chunk  Whether to output audio mel chuncks, which is divided by video fps

```


API:

Please refer to function `extract_audio_from_video`.

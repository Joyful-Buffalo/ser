import json
import os

import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB


class LogMelSpectrogram(object):
    """Returns the log-mel spectrogram of the resampled audio clip.

    Args:
        sample_rate (int, optional): Sample rate to which the audio clip
            will be resampled. Default: 16000
        audio_length (int, optional): Length in seconds of the audio
            clip, if the clip is longer it will be cut to this
            value, if it is shorter it will be padded with zeros.
            Default: 8
        n_fft (int): Size of FFT for the spectrogram. Default: 2048
        hop_length (int, optional): Length of hop between STFT windows.
            Default: 512
    """

    def __init__(self, sample_rate, audio_length, n_fft, hop_length):
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, waveform):
        waveform = Resample(new_freq=self.sample_rate)(waveform)

        # cut or pad audio clip
        waveform.resize_(self.sample_rate * self.audio_length)

        mel_spec = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )(waveform)

        # since we pad the audio clip the mel-spectrogram might have
        # nan values, compromising the loss computation
        mel_spec = torch.nan_to_num(mel_spec, 1e-5)

        log_mel_spec = AmplitudeToDB()(mel_spec)

        return log_mel_spec


def get_path():
    json_path = '/home/pwy/lhc/HHpaper/dataset/IEMOCAP/labels.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    # print(data['neu'], '\n', type(data['neu']))
    print(data.keys())
    return data


def main():
    savdir = '/home/pwy/lhc/HHpaper/dataset/IEMOCAP/log-mel/'
    if not os.path.exists(savdir):
        os.mkdir(savdir)
    count = 0
    json_data = get_path()
    transform = LogMelSpectrogram(sample_rate=16000, audio_length=3, n_fft=2048, hop_length=512)
    for key in json_data.keys():
        num = 0
        for wav_path in json_data[key]:
            signal, _ = torchaudio.load(wav_path)
            signal = torch.mean(signal, 0)
            log_mel = transform(signal)

            # log_mel_np = log_mel.numpy()
            # log_mel_normalized = (log_mel_np - log_mel_np.min()) / (log_mel_np.max() - log_mel_np.min()) * 255
            # log_mel_image = Image.fromarray(log_mel_normalized.astype('uint8'), mode='L')
            plt.imshow(log_mel)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            break
        # break
    print(count)


if __name__ == '__main__':
    main()

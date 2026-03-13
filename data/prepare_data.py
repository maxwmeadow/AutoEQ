import os
import numpy as np
import pandas as pd
import librosa
from pedalboard import Pedalboard, PeakFilter

RAW_DIRECTORY = 'data/raw/VocalSet1-2/data_by_singer'
SPECTROGRAM_DIRECTORY = 'data/processed/spectrograms'
LABELS_FILE = 'data/processed/labels.csv'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
CLIP_FRAMES = 431

EQ_BANDS = {
    'low':  {'freq': (60, 300),  'gain': (-8, 8), 'q': (0.5, 2.0)},
    'mid':  {'freq': (300, 3000), 'gain': (-6, 6), 'q': (0.5, 4.0)},
    'high': {'freq': (3000, 16000),'gain': (-8, 8), 'q': (0.5, 2.0)},
}

def random_eq_parameters():
    parameters = {}
    for band, ranges in EQ_BANDS.items():
        parameters[f'freq_{band}'] = np.random.uniform(*ranges['freq'])
        parameters[f'gain_{band}'] = np.random.uniform(*ranges['gain'])
        parameters[f'q_{band}'] = np.random.uniform(*ranges['q'])
    return parameters

def apply_random_eq(audio, sample_rate, parameters):
    board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=parameters['freq_low'], gain_db=parameters['gain_low'], q=parameters['q_low']),
        PeakFilter(cutoff_frequency_hz=parameters['freq_mid'], gain_db=parameters['gain_mid'], q=parameters['q_mid']),
        PeakFilter(cutoff_frequency_hz=parameters['freq_high'], gain_db=parameters['gain_high'], q=parameters['q_high']),
    ])
    return board(audio, sample_rate)

def audio_to_log_mel_spectrogram(audio, sample_rate):
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < CLIP_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, CLIP_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :CLIP_FRAMES]
    return log_mel[np.newaxis, :, :].astype(np.float32)

def chunk_audio(audio):
    clips = []
    for start in range(0, len(audio), CLIP_SAMPLES):
        clip = audio[start:start + CLIP_SAMPLES]
        if np.sqrt(np.mean(clip**2)) > 0.001:
            clips.append(clip)

    return clips

def prepare():
    os.makedirs(SPECTROGRAM_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)

    rows = []
    clip_id = 0

    for root, directories, files in os.walk(RAW_DIRECTORY):
        valid_directories = []
        for directory in sorted(directories):
            if not directory.startswith('.'):
                valid_directories.append(directory)
                
        directories[:] = valid_directories

        for filename in sorted(files):
            if not filename.endswith('.wav'):
                continue

            wav_path = os.path.join(root, filename)
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            clips = chunk_audio(audio)

            for clip in clips:
                parameters = random_eq_parameters()
                wrecked = apply_random_eq(clip, SAMPLE_RATE, parameters)
                clean_spectrogram   = audio_to_log_mel_spectrogram(clip, SAMPLE_RATE)
                wrecked_spectrogram = audio_to_log_mel_spectrogram(wrecked, SAMPLE_RATE)
                spectrogram = wrecked_spectrogram - clean_spectrogram

                npy_name = f'{clip_id:06d}.npy'
                np.save(os.path.join(SPECTROGRAM_DIRECTORY, npy_name), spectrogram)

                rows.append({
                    'id': npy_name,
                    'freq_low': parameters['freq_low'],
                    'gain_low': parameters['gain_low'],
                    'q_low': parameters['q_low'],
                    'freq_mid': parameters['freq_mid'],
                    'gain_mid': parameters['gain_mid'],
                    'q_mid': parameters['q_mid'],
                    'freq_high': parameters['freq_high'],
                    'gain_high': parameters['gain_high'],
                    'q_high': parameters['q_high'],
                })
                clip_id += 1

            if clip_id % 100 == 0 and clip_id > 0:
                print(f'{clip_id} clips processed')

    pd.DataFrame(rows).to_csv(LABELS_FILE, index=False)
    print(f"Labels saved to {LABELS_FILE} with {len(rows)} entries.")

if __name__ == "__main__":
    prepare()
        
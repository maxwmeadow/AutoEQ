import os
import numpy as np
import pandas as pd
import librosa
from pedalboard import Pedalboard, PeakFilter
import argparse
from multiprocessing import Pool

os.environ.setdefault('MEDLEYDB_PATH', 'data/')
import medleydb as mdb

RAW_DIRECTORY = 'data/raw/VocalSet1-2/data_by_singer'
MEDLEYDB_V2_DIRECTORY = 'data/V2'

SAMPLE_RATE = 22050
HOP_LENGTH = 512
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
CLIP_FRAMES = 431

EQ_BANDS = {
    'low':  {'freq': (60, 300),  'gain': (-8, 8), 'q': (0.5, 2.0)},
    'mid':  {'freq': (300, 3000), 'gain': (-6, 6), 'q': (0.5, 4.0)},
    'high': {'freq': (3000, 16000),'gain': (-8, 8), 'q': (0.5, 2.0)},
}

VOCAL_LABELS = {'male singer', 'female singer', 'vocalists'}

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

def audio_to_log_mel_spectrogram(audio, sample_rate, n_mels):
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=HOP_LENGTH, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    frames = log_mel.shape[1]

    if frames < CLIP_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, CLIP_FRAMES - frames)))
    else:
        log_mel = log_mel[:, :CLIP_FRAMES]

    return log_mel[np.newaxis, :, :].astype(np.float32)

def audio_to_cqt_spectrogram(audio, sample_rate):
    cqt = librosa.cqt(audio, sr=sample_rate, hop_length=HOP_LENGTH, n_bins=128, bins_per_octave=24)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    frames = log_cqt.shape[1]

    if frames < CLIP_FRAMES:
        log_cqt = np.pad(log_cqt, ((0, 0), (0, CLIP_FRAMES - frames)))
    else:
        log_cqt = log_cqt[:, :CLIP_FRAMES]

    return log_cqt[np.newaxis, :, :].astype(np.float32)

def chunk_audio(audio):
    clips = []
    for start in range(0, len(audio), CLIP_SAMPLES):
        clip = audio[start:start + CLIP_SAMPLES]
        if np.sqrt(np.mean(clip**2)) > 0.001:
            clips.append(clip)

    return clips

def get_vocalset_files():
    wav_files = []
    for root, directories, files in os.walk(RAW_DIRECTORY):
        directories[:] = sorted(directory for directory in directories if not directory.startswith('.'))
        for filename in sorted(files):
            if filename.endswith('.wav'):
                wav_files.append(os.path.join(root, filename))

    return wav_files

def get_medleydb_files():
    tracks = list(mdb.load_multitracks(mdb.TRACK_LIST_V2))
    wav_files = []
    for track in tracks:
        for stem_id, stem in track.stems.items():
            if set(stem.instrument) & VOCAL_LABELS:
                path = os.path.join(MEDLEYDB_V2_DIRECTORY, track.track_id, f'{track.track_id}_STEMS', f'{track.track_id}_STEM_{stem_id:02d}.wav')
                if os.path.exists(path):
                    wav_files.append(path)

    return wav_files

def process_file(args):
    file_index, wav_file, spectrogram_directory, mode = args
    audio, _ = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
    clips = chunk_audio(audio)

    rows = []
    for clip_index, clip in enumerate(clips):
        parameters = random_eq_parameters()
        wrecked = apply_random_eq(clip, SAMPLE_RATE, parameters)

        if mode == 'mel256':
            clean_spectrogram = audio_to_log_mel_spectrogram(clip, SAMPLE_RATE, 256)
            wrecked_spectrogram = audio_to_log_mel_spectrogram(wrecked, SAMPLE_RATE, 256)
        else:
            clean_spectrogram = audio_to_cqt_spectrogram(clip, SAMPLE_RATE)
            wrecked_spectrogram = audio_to_cqt_spectrogram(wrecked, SAMPLE_RATE)

        spectrogram = wrecked_spectrogram - clean_spectrogram
        npy_name = f'{file_index:05d}_{clip_index:04d}.npy'
        np.save(os.path.join(spectrogram_directory, npy_name), spectrogram)

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

    return rows

def prepare(mode, dataset, num_workers):
    output_dir = f'data/processed_{mode}_{dataset}'
    spectrogram_directory = os.path.join(output_dir, 'spectrograms')
    labels_file = os.path.join(output_dir, 'labels.csv')
    os.makedirs(spectrogram_directory, exist_ok=True)

    if dataset == 'vocalset':
        wav_files = get_vocalset_files()
    elif dataset == 'medleydb':
        wav_files = get_medleydb_files()
    else:
        wav_files = get_vocalset_files() + get_medleydb_files()

    args = [(i, path, spectrogram_directory, mode) for i, path in enumerate(wav_files)]
    all_rows = []

    with Pool(num_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_file, args, chunksize=1)):
            all_rows.extend(rows)
            if (i + 1) % 10 == 0:
                print(f'Processed {i + 1}/{len(wav_files)} files of {len(all_rows)}')

    pd.DataFrame(all_rows).to_csv(labels_file, index=False)
    print(f"Labels saved to {labels_file} with {len(all_rows)} entries.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['mel256', 'cqt'])
    parser.add_argument('--dataset', type=str, required=True, choices=['vocalset', 'medleydb', 'combined'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prepare(args.mode, args.dataset, args.workers)
        
from scipy.io import wavfile
import os
import numpy as np
from tqdm import tqdm
import json
import sys
from datetime import datetime, timedelta

# Utility functions for AudioSeg


def GetTime(video_seconds):
    if (video_seconds < 0):
        return 00

    else:
        sec = timedelta(seconds=float(video_seconds))
        d = datetime(1, 1, 1) + sec

        instant = str(d.hour).zfill(2) + ':' + str(d.minute).zfill(2) + \
            ':' + str(d.second).zfill(2) + str('.001')

        return instant


def GetTotalTime(video_seconds):
    sec = timedelta(seconds=float(video_seconds))
    d = datetime(1, 1, 1) + sec
    delta = str(d.hour) + ':' + str(d.minute) + ":" + str(d.second)

    return delta


def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]


def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))


def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1


def run_audioseg(input_file=None, output_dir=None, min_silence_length=0.6, silence_threshold=1e-4, step_duration=0.03/10, max_slice_length=30.0):
    '''
    Last Acceptable Values

    min_silence_length = 0.3
    silence_threshold = 1e-3
    step_duration = 0.03/10

    '''
    if not input_file:
        raise ValueError("No input file selected.")
    if not output_dir:
        raise ValueError("No output directory selected.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_filename = input_file
    window_duration = min_silence_length
    if step_duration is None:
        step_duration = window_duration / 10.
    else:
        step_duration = step_duration

    output_filename_prefix = os.path.splitext(
        os.path.basename(input_filename))[0]
    dry_run = False

    sample_rate, samples = wavfile.read(filename=input_filename, mmap=True)
    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])

    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)
    max_slice_size = int(max_slice_length * sample_rate)

    signal_windows = windows(
        signal=samples, window_size=window_size, step_size=step_size)
    window_energy = (energy(w) / max_energy for w in tqdm(signal_windows,
                     total=int(len(samples) / float(step_size))))
    window_silence = (e > silence_threshold for e in window_energy)
    cut_times = [0] + [r * step_duration for r in rising_edges(window_silence)] + [
        len(samples) / sample_rate]

    cut_samples = [int(t * sample_rate) for t in cut_times]

    cut_ranges = []
    for i in range(len(cut_samples) - 1):
        start = cut_samples[i]
        stop = cut_samples[i+1]
        while start < stop:
            cut_ranges.append(
                (len(cut_ranges), start, min(start + max_slice_size, stop)))
            start += max_slice_size

    video_sub = {str(i): [GetTime(start/sample_rate), GetTime(stop/sample_rate)]
                 for i, start, stop in cut_ranges}

    for i, start, stop in tqdm(cut_ranges):
        output_file_path = "{}_{:03d}.wav".format(
            os.path.join(output_dir, output_filename_prefix), i)
        if not dry_run:
            wavfile.write(filename=output_file_path,
                          rate=sample_rate, data=samples[start:stop])

    with open(os.path.join(output_dir, output_filename_prefix + '.json'), 'w') as output:
        json.dump(video_sub, output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audiosplitter.py <input_path> <output_path> [<min_silence_length> <silence_threshold> <step_duration> <max_slice_length>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Opsiyonel argümanları ayarla
    min_silence_length = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    silence_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    step_duration = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03/10
    max_slice_length = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0

    run_audioseg(input_path, output_path, min_silence_length, silence_threshold, step_duration, max_slice_length)
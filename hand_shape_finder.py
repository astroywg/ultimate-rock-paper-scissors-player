import torchvision, torch
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def get_freq(audio_clip, audio_fps):
    size = len(audio_clip)
    audio_freq = np.fft.fft(audio_clip)[:size//2] / size
    audio_freq[1:] *= 2

    # to get more accurate peak frequency
    mp = 15 # multiplier
    sd = size * mp # size dense
    audio_freq_dense = np.zeros(sd // 2)
    audio_freq_dense[::mp] = np.abs(audio_freq)

    audio_freq_dense = gaussian_filter1d(audio_freq_dense, sigma=10*mp)

    freqs = np.fft.fftfreq(sd, 1/audio_fps)[:sd//2]
    peak_freq = freqs[np.argmax(audio_freq_dense)]
    return peak_freq


def check_valid_d2t(d_intervals):
    return all((
        0 <= np.max(d_intervals) < 0.1,
        -0.1 < np.min(d_intervals) <= 0
    ))


def read_find_hands(video_name, start_time, end_time, **kwargs):

    video, audio, metadata = torchvision.io.read_video(video_name)

    settings = {
        'peak_t_tolerance': 0.07,
        'peak_height': 2e-4
    }

    settings.update(kwargs)

    audio_fps = metadata['audio_fps']
    assert audio_fps == 48000

    start_point = audio_fps * start_time
    end_point =audio_fps * end_time

    t = np.arange(start_point, end_point) / audio_fps
    clip = audio[0, start_point:end_point] ** 2
    clip = clip.numpy()
    clip = gaussian_filter1d(clip, 1000)

    peaks, peak_properties = find_peaks(clip, distance=int(audio_fps * 0.75), height=settings['peak_height'])
    peak_t = t[peaks]

    peak_i = np.arange(len(peak_t))
    fit = np.polynomial.polynomial.polyfit(peak_i, peak_t, deg=1)
    linear_peak_t = peak_i * fit[1] + fit[0]
    linear_error = np.abs(linear_peak_t - peak_t)

    # substitute peaks which error is bigger than tolerance
    bad_peaks = linear_error > settings['peak_t_tolerance']
    peak_t[bad_peaks] = linear_peak_t[bad_peaks]

    assert 1.01 < fit[1] < 1.03
    assert check_valid_d2t(np.diff(peak_t, n=2))

    audio_points = peak_t * audio_fps

    offset = audio_fps // 30
    freqs = []
    for audio_point in audio_points:
        aud_mid_point = round(audio_point)
        aud_start_point = aud_mid_point - offset
        aud_end_point = aud_mid_point + offset
        aud_clip = audio[0, aud_start_point:aud_end_point]
        freqs.append(get_freq(aud_clip.numpy(), audio_fps))

    freqs = np.array(freqs)

    spam = 0
    for i in range(3):
        high_note = np.all(freqs[i::3] > 600) 
        low_note = np.all(freqs[i::3] < 600)
        assert high_note or low_note
        if high_note:
            spam = i

    peak_frames = np.round(peak_t * metadata['video_fps']).astype(np.int64)
    hand_shape_frames = peak_frames[spam::3]

    return video, hand_shape_frames


if __name__ == '__main__':
    filename = 'dataset/paper.mp4'
    video, hand_shape_frames = read_find_hands(filename, 20, 600, peak_height=2e-4)
    np.save(filename + '.npy', hand_shape_frames)
    print(np.diff(hand_shape_frames))

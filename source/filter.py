import numpy as np
import soundfile as sf
from scipy.fft import fft, ifft

INT16_MAX=32768
PLOT_LEN=200

def filter_spectrum(spectrum: np.ndarray) -> np.ndarray:
    amplitude = np.abs(spectrum)
    max_amp = np.max(amplitude)
    spectrum_filtered = spectrum.copy()
    spectrum_filtered[amplitude < max_amp / 2.0] = 0
    return spectrum_filtered


def filter_file(input_path, output_path, ax = None):
    data, rate = sf.read(input_path)
    data_float = data.astype(np.float32) / INT16_MAX

    spectrum = fft(data_float)
    filtered_spectrum = filter_spectrum(spectrum)
    cleaned = np.real(ifft(filtered_spectrum))
    cleaned /= np.max(np.abs(cleaned) + 1e-12)
    cleaned_normalised = (cleaned * INT16_MAX).astype(np.int16)

    sf.write(output_path, cleaned_normalised, rate, subtype='PCM_16')

    if ax is not None:
        ax.plot(cleaned_normalised[:PLOT_LEN])
        ax.grid()


def make_plot(input_path, ax):
    data, rate = sf.read(input_path)
    ax.plot(data[:PLOT_LEN])
    ax.grid()


def get_random_noise(t, base_amplitude, amount=1000):
    noise = np.zeros_like(t)
    rng = np.random.default_rng()

    for _ in range(amount):
        freq = rng.uniform(-2000, 2000)
        phase = rng.uniform(0, 2000)
        noise += 0.5 * base_amplitude * np.sin(2 * np.pi * freq * t + phase)

    return noise


def get_white_noise(t, base_amplitude):
    rng = np.random.default_rng()
    noise = rng.uniform(-base_amplitude / 2, base_amplitude / 2, size=t.shape)
    return noise


def make_noisy(data_float, rate):
    t = np.arange(len(data_float)) / rate
    base_amplitude = np.median(np.abs(data_float))

    # noise = get_random_noise(t, base_amplitude)
    noise = get_white_noise(t, base_amplitude)

    noised = data_float + noise

    noised /= np.max(np.abs(noised) + 1e-12)
    noised_normalised = (noised * INT16_MAX).astype(np.int16)

    return noised_normalised


def noise_file(input_path, output_path, ax=None):
    data, rate = sf.read(input_path)
    data_float = data.astype(np.float32) / INT16_MAX

    noised = make_noisy(data_float, rate)
    sf.write(output_path, noised, rate, subtype='PCM_16')

    if ax is not None:
        ax.plot(noised[:PLOT_LEN])
        ax.grid()


def npy_to_wav(input_path, output_path, rate):
    data = np.load(input_path)
    sf.write(output_path, data, rate, subtype='PCM_16')



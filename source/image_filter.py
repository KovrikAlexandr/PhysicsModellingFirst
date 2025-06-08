import numpy as np
from PIL import Image


def fft2d(image):
    return np.fft.fft2(image)



def ifft2d(spectrum):
    return np.fft.ifft2(spectrum)



def fftshift2d(spectrum):
    return np.fft.fftshift(spectrum)



def ifftshift2d(spectrum):
    return np.fft.ifftshift(spectrum)



def save_bmp(filename, data):
    img = Image.fromarray(data.astype(np.uint8))
    img.save(filename)



def create_test_image(path, width=512, height=512):
    x = np.linspace(0, 2 * np.pi * 5, width)
    y = np.linspace(0, 2 * np.pi * 5, height)
    xx, yy = np.meshgrid(x, y)

    zz = 127.5 + 127.5 * np.sin(xx / 5) * np.cos(yy / 5)

    save_bmp(path, zz.astype(np.uint8))



def noise_image_sin(input, output):
    image = Image.open(input).convert("L")
    img_array = np.array(image).astype(np.float32)

    HEIGHT, WIDTH = img_array.shape
    rng = np.random.default_rng()
    noise_pattern = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for _ in range(5):
        fx = rng.uniform(5, 30)
        fy = rng.uniform(5, 30)
        phase = rng.uniform(0, 2 * np.pi)

        y = np.arange(HEIGHT).reshape(-1, 1)
        x = np.arange(WIDTH).reshape(1, -1)
        noise_pattern += np.sin(2 * np.pi * (fx * x / WIDTH + fy * y / HEIGHT) + phase)

    noisy = img_array + 15.0 * noise_pattern / 5.0
    save_bmp(output, np.clip(noisy, 0, 255))



def noise_image_white_noise(input, output):
    image = Image.open(input).convert("L")
    img_array = np.array(image).astype(np.float32)

    max_amp = np.max(img_array)
    noise_amp = max_amp / 2

    rng = np.random.default_rng()
    noise = rng.normal(0, 1, size=img_array.shape)
    noise = noise / np.max(np.abs(noise)) * noise_amp

    noisy = img_array + noise
    save_bmp(output, np.clip(noisy, 0, 255))



def noise_image_salt(input, output):
    image = Image.open(input).convert("L")
    img_array = np.array(image).astype(np.float32)

    HEIGHT, WIDTH = img_array.shape
    rng = np.random.default_rng()

    prob = 0.01
    mask = rng.random((HEIGHT, WIDTH))

    salt_pepper = img_array.copy()
    salt_pepper[mask < prob / 2] = 0
    salt_pepper[mask > 1 - prob / 2] = 255

    save_bmp(output, salt_pepper.astype(np.uint8))




def filter_image(input_path, output_path):
    image = Image.open(input_path).convert("L")
    data = np.array(image, dtype=np.float32)

    spectrum = np.fft.fft2(data)
    spectrum_shifted = np.fft.fftshift(spectrum)

    amplitude = np.abs(spectrum_shifted)
    threshold = np.max(amplitude) / 10
    spectrum_shifted[amplitude < threshold] = 0

    spectrum_unshifted = np.fft.ifftshift(spectrum_shifted)
    filtered = np.fft.ifft2(spectrum_unshifted).real

    save_bmp(output_path, np.clip(filtered, 0, 255).astype(np.uint8))

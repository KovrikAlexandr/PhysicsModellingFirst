from os.path import exists

from source import filter
from source import image_filter
import matplotlib.pyplot as plt
import os


INPUT_FILE="sound/signal.wav"
CLEANED_FILE="sound/clean.wav"
NOISED_FILE="sound/noised.wav"
CLEANED_AGAIN_FILE="sound/cleaned_again.wav"

INPUT_IMG="images/test.bmp"
NOISED_IMG="images/noised.bmp"
CLEANED_IMG="images/cleaned.bmp"


if __name__ == '__main__':
    # Images
    os.makedirs('images', exist_ok=True)
    image_filter.create_test_image(INPUT_IMG)
    image_filter.noise_image_white_noise(INPUT_IMG, NOISED_IMG)
    image_filter.filter_image(NOISED_IMG, CLEANED_IMG)


    # SOUNDS
    filter.npy_to_wav('sound/signal.npy', 'sound/signal.wav', 9000)

    fig, axs = plt.subplots(2, 2)

    # Raw sound
    filter.make_plot(INPUT_FILE, axs[0, 0])
    axs[0, 0].set_title("Raw")

    # Clean
    filter.filter_file(INPUT_FILE, CLEANED_FILE, axs[0, 1])
    axs[0, 1].set_title("Cleaned")

    # Noised
    filter.noise_file(CLEANED_FILE, NOISED_FILE, axs[1, 0])
    axs[1, 0].set_title("Noised")

    # Cleaned again
    filter.filter_file(NOISED_FILE, CLEANED_AGAIN_FILE, axs[1, 1])
    axs[1, 1].set_title("Cleaned again")

    plt.show()




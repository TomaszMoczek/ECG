#!/usr/bin/python3
import os
import pygame
import configparser

from samples import Samples
from dspplotter import DspPlotter


def plot_signal(fs: int, data: tuple, labels: tuple, file_name: str) -> None:
    global is_signal
    global is_sound
    global is_spectrogram

    if is_signal:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

        Samples.write_wave_file(fs=fs, data=data, file_path=file_path)

        if is_sound:
            pygame.mixer.init(frequency=fs)
            pygame.mixer.music.load(filename=file_path)
            pygame.mixer.music.play(loops=-1)

        dsp_plotter = DspPlotter()

        dsp_plotter.plot(
            fs=fs,
            data=data,
            labels=labels,
            div_by_N=True,
            log_freq=True,
            block=False if is_spectrogram else True,
        )

        if is_spectrogram:
            dsp_plotter.spectrogram(
                fs=fs,
                data=data,
                labels=labels,
                segmentsize=8,
                log_freq=True,
                vmin=-130,
            )

        if is_sound:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()


def main() -> None:
    samples = Samples()

    fs = samples.get_fs()
    data = samples.get_data()

    plot_signal(fs=fs, data=data, labels=("MLII", "V1"), file_name="samples.wav")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    is_signal = config.getboolean("default", "signal")
    is_sound = config.getboolean("default", "sound")
    is_spectrogram = config.getboolean("default", "spectrogram")

    main()

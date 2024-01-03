#!/usr/bin/python3
import os
import pygame
import configparser

from samples import Samples
from dspplotter import DspPlotter


def main() -> None:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    is_signal = config.getboolean("default", "signal")
    is_sound = config.getboolean("default", "sound")
    is_spectrogram = config.getboolean("default", "spectrogram")

    samples = Samples()
    dsp_plotter = DspPlotter()

    fs = samples.get_fs()
    data = samples.get_data()

    labels = ("MLII", "V1")

    if is_signal:
        samples.write_wave_file()

        if is_sound:
            pygame.mixer.init(frequency=fs)
            pygame.mixer.music.load(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
            )
            pygame.mixer.music.play(loops=-1)

        dsp_plotter.plot(
            data=data,
            labels=labels,
            fs=fs,
            div_by_N=True,
            freq_lim=(0, fs / 2),
        )

        if is_spectrogram:
            dsp_plotter.spectrogram(
                data=data,
                labels=labels,
                fs=fs,
                segmentsize=8,
            )

        if is_sound:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()


if __name__ == "__main__":
    main()

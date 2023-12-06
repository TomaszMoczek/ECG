#!/usr/bin/python3
import os
import pygame
import configparser

from samples import Samples
from dspplotter import DspPlotter


def main() -> None:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    is_sound = config.getboolean("default", "sound")
    is_spectrogram = config.getboolean("default", "spectrogram")

    samples = Samples()
    fs = samples.get_fs()
    data = samples.get_data()
    samples.write_wave_file()

    if is_sound:
        pygame.mixer.init(frequency=fs)
        pygame.mixer.music.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
        )
        pygame.mixer.music.play(loops=-1)

    dsp_plotter = DspPlotter()
    dsp_plotter.plot(
        data=data, labels=("MLII", "V1"), fs=fs, freqresp=True, phaseresp=True
    )

    if is_spectrogram:
        dsp_plotter.wavplot(
            # data=data[0],
            # fs=360
            data=r"/home/tomasz/Projects/Console/audio_high_quality.wav"
        )

    if is_sound:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()


if __name__ == "__main__":
    main()

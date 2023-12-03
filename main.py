#!/usr/bin/python3
import os
import pygame

from samples import Samples
from dspplotter import DspPlotter


def main() -> None:
    samples = Samples()
    fs = samples.get_fs()
    data = samples.get_data()
    samples.write_wave_file()
    pygame.mixer.init(frequency=fs)
    pygame.mixer.music.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
    )
    pygame.mixer.music.play(loops=-1)
    dsp_plotter = DspPlotter()
    dsp_plotter.plot(data=data, labels=("MLII", "V1"), fs=fs, freqresp=False)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()


if __name__ == "__main__":
    main()

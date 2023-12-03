#!/usr/bin/python3
from samples import Samples
from dspplotter import DspPlotter
import pygame
import os


def init_wave_player(fs: int) -> None:
    pygame.mixer.init(frequency=fs)


def play_wave_file(file_path: str) -> None:
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play(loops=-1)


def stop_wave_file() -> None:
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()


def main() -> None:
    samples = Samples()
    fs = samples.get_fs()
    mlii, v1 = samples.get_data()
    samples.write_wave_file()
    init_wave_player(fs=fs)
    play_wave_file(
        file_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "samples.wav"
        )
    )
    dsp_plotter = DspPlotter()
    dsp_plotter.plot(data=mlii, title="MLII", Fs=fs, freqresp=False)
    dsp_plotter.plot(data=v1, title="V1", Fs=fs, freqresp=False)
    stop_wave_file()


if __name__ == "__main__":
    main()

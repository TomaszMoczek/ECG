#!/usr/bin/python3
from ecg import Ecg
from dspplotter import DspPlotter
import pygame
import numpy
import wave
import time
import os


def write_wave_file(data: list, fs: int, file_path: str) -> None:
    audio = numpy.array([data, data]).T
    audio = (audio * (2**15 - 1)).astype("<h")
    with wave.open(file_path, "w") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(fs)
        file.writeframes(audio.tobytes())


def init_wave_player(fs: int) -> None:
    pygame.mixer.init(frequency=fs)


def play_wave_file(file_path: str) -> None:
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def stop_wave_file() -> None:
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.unload()


def main() -> None:
    ecg = Ecg()
    fs = ecg.get_fs()
    mlii, v1 = ecg.get_data()
    write_wave_file(
        data=mlii,
        fs=fs,
        file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlii.wav"),
    )
    write_wave_file(
        data=v1,
        fs=fs,
        file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1.wav"),
    )
    dsp_plotter = DspPlotter()
    init_wave_player(fs=fs)
    play_wave_file(
        file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlii.wav")
    )
    dsp_plotter.plot(data=mlii, title="MLII", Fs=fs, freqresp=False)
    stop_wave_file()
    time.sleep(1.0)
    play_wave_file(
        file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1.wav")
    )
    dsp_plotter.plot(data=v1, title="V1", Fs=fs, freqresp=False)
    stop_wave_file()


if __name__ == "__main__":
    main()

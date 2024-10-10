#!/usr/bin/python3
import os
import numpy
import pygame
import configparser

from scipy import signal

from samples import Samples
from dspplotter import DspPlotter


def plot_signal(fs: int, data: numpy.ndarray, labels: tuple, file_name: str) -> None:
    global is_sound
    global is_spectrogram

    dsp_plotter = DspPlotter()

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    Samples.write_wave_file(fs=fs, data=data, file_path=file_path)

    if is_sound:
        pygame.mixer.init(frequency=fs)
        pygame.mixer.music.load(filename=file_path)
        pygame.mixer.music.play(loops=-1)

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
    global is_signal

    samples = Samples()

    fs = samples.get_fs()
    data = samples.get_data()

    if is_signal:
        plot_signal(fs=fs, data=data, labels=("MLII", "V1"), file_name="samples.wav")

        mlii = signal.wiener(im=data[0])
        v1 = signal.wiener(im=data[1])

        plot_signal(
            fs=fs,
            data=numpy.vstack((mlii, v1)),
            labels=("MLII [Wiener filtered]", "V1 [Wiener filtered]"),
            file_name="samples-wiener-filtered.wav",
        )

    w0 = 60.0
    bw = 2.0
    b, a = signal.iirnotch(w0=w0, Q=w0 / bw, fs=float(fs))
    x = signal.unit_impulse(shape=128)
    y = signal.lfilter(b=b, a=a, x=x)

    dsp_plotter = DspPlotter()

    dsp_plotter.plot(
        fs=fs,
        data=numpy.vstack((y,)),
        labels=("IIR notch filter 60 Hz",),
        log_freq=True,
        phaseresp=True,
    )


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    is_signal = config.getboolean("default", "signal")
    is_sound = config.getboolean("default", "sound")
    is_spectrogram = config.getboolean("default", "spectrogram")
    main()

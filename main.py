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

    Samples.write_wave_file(
        fs=fs,
        data=data,
        file_path=file_path,
    )

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
        plot_signal(
            fs=fs,
            data=data,
            labels=("MLII", "V1"),
            file_name="samples.wav",
        )

        mlii = signal.wiener(im=data[0])
        v1 = signal.wiener(im=data[1])

        plot_signal(
            fs=fs,
            data=numpy.vstack((mlii, v1)),
            labels=("MLII [Wiener filtered]", "V1 [Wiener filtered]"),
            file_name="samples-wiener-filtered.wav",
        )

    polynomials: list[dict] = []
    x = signal.unit_impulse(shape=128)
    w0 = 60.0
    bw = 2.0
    b, a = signal.iirnotch(w0=w0, Q=w0 / bw, fs=float(fs))
    polynomials.append({"b": b, "a": a})
    wn = 30.0
    b, a = signal.butter(N=4, Wn=wn, btype="lowpass", fs=float(fs))
    polynomials.append({"b": b, "a": a})
    wn = 0.5
    b, a = signal.butter(N=7, Wn=wn, btype="highpass", fs=float(fs))
    polynomials.append({"b": b, "a": a})
    for i in range(0, len(polynomials)):
        if i == 0:
            y1 = signal.lfilter(b=polynomials[i]["b"], a=polynomials[i]["a"], x=x)
        elif i == 1:
            y2 = signal.lfilter(b=polynomials[i]["b"], a=polynomials[i]["a"], x=x)
        elif i == 2:
            y3 = signal.lfilter(b=polynomials[i]["b"], a=polynomials[i]["a"], x=x)
        y4 = signal.lfilter(
            b=polynomials[i]["b"], a=polynomials[i]["a"], x=x if i == 0 else y4
        )

    dsp_plotter = DspPlotter()

    dsp_plotter.plot(
        fs=fs,
        data=numpy.vstack((y1, y2, y3, y4)),
        labels=(
            "IIR notch 60 Hz",
            "IIR low-pass 30 Hz",
            "IIR high-pass 0.5 Hz",
            "IIR cascaded",
        ),
        phaseresp=True,
    )


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    is_signal = config.getboolean("default", "signal")
    is_sound = config.getboolean("default", "sound")
    is_spectrogram = config.getboolean("default", "spectrogram")
    main()

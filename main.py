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
    global is_iir
    global is_signal

    samples = Samples()
    dsp_plotter = DspPlotter()

    fs = samples.get_fs()
    data = samples.get_data()

    x = signal.unit_impulse(shape=128)

    if is_signal:
        plot_signal(
            fs=fs,
            data=data,
            labels=("MLII", "V1"),
            file_name="samples.wav",
        )

        data[0] = signal.wiener(im=data[0])
        data[1] = signal.wiener(im=data[1])

        plot_signal(
            fs=fs,
            data=data,
            labels=("MLII [Wiener filtered]", "V1 [Wiener filtered]"),
            file_name="samples-wiener-filtered.wav",
        )

    if is_iir:
        b, a = signal.iirnotch(w0=60.0, Q=(60.0 / 2.0), fs=float(fs))
        y1 = signal.lfilter(b=b, a=a, x=x)
        y4 = signal.lfilter(b=b, a=a, x=x)
        data[0] = signal.filtfilt(b=b, a=a, x=data[0])
        data[1] = signal.filtfilt(b=b, a=a, x=data[1])
    else:
        h = signal.firwin(numtaps=231, cutoff=[59.0, 61.0], fs=float(fs))
        y1 = signal.lfilter(b=h, a=1, x=x)
        y4 = signal.lfilter(b=h, a=1, x=x)
        data[0] = signal.filtfilt(b=h, a=1, x=data[0])
        data[1] = signal.filtfilt(b=h, a=1, x=data[1])

    if is_iir:
        sos = signal.butter(N=2, Wn=30.0, btype="lowpass", output="sos", fs=float(fs))
        y2 = signal.sosfilt(sos=sos, x=x)
        y4 = signal.sosfilt(sos=sos, x=y4)
        data[0] = signal.sosfiltfilt(sos=sos, x=data[0])
        data[1] = signal.sosfiltfilt(sos=sos, x=data[1])
    else:
        h = signal.firwin(numtaps=9, cutoff=30.0, fs=float(fs))
        y2 = signal.lfilter(b=h, a=1, x=x)
        y4 = signal.lfilter(b=h, a=1, x=y4)
        data[0] = signal.filtfilt(b=h, a=1, x=data[0])
        data[1] = signal.filtfilt(b=h, a=1, x=data[1])

    if is_iir:
        sos = signal.butter(N=2, Wn=0.5, btype="highpass", output="sos", fs=float(fs))
        y3 = signal.sosfilt(sos=sos, x=x)
        y4 = signal.sosfilt(sos=sos, x=y4)
        # data[0] = signal.sosfiltfilt(sos=sos, x=data[0])
        # data[1] = signal.sosfiltfilt(sos=sos, x=data[1])
    else:
        sos = signal.butter(N=2, Wn=0.5, btype="highpass", output="sos", fs=float(fs))
        y3 = signal.sosfilt(sos=sos, x=x)
        y4 = signal.sosfilt(sos=sos, x=y4)
        # data[0] = signal.sosfiltfilt(sos=sos, x=data[0])
        # data[1] = signal.sosfiltfilt(sos=sos, x=data[1])

    dsp_plotter.plot(
        fs=fs,
        data=numpy.vstack((y1, y2, y3, y4)),
        labels=(
            "IIR notch 60 Hz" if is_iir else "FIR band-stop 60 Hz",
            "IIR low-pass 30 Hz" if is_iir else "FIR low-pass 30 Hz",
            "IIR high-pass 0.5 Hz" if is_iir else "FIR high-pass 0.5 Hz",
            "IIR cascaded" if is_iir else "FIR cascaded",
        ),
        phaseresp=True,
    )

    if is_signal:
        plot_signal(
            fs=fs,
            data=data,
            labels=(
                "MLII [IIR filtered]" if is_iir else "MLII [FIR filtered]",
                "V1 [IIR filtered]" if is_iir else "V1 [FIR filtered]",
            ),
            file_name=(
                "samples-iir-filtered.wav" if is_iir else "samples-fir-filtered.wav"
            ),
        )


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    is_iir = config.getboolean("default", "iir")
    is_sound = config.getboolean("default", "sound")
    is_signal = config.getboolean("default", "signal")
    is_spectrogram = config.getboolean("default", "spectrogram")
    main()

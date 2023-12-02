#!/usr/bin/python3
from ecg import Ecg
from dspplotter import DspPlotter
import numpy
import wave
import os


def write_wave_file(data: list, fs: int, file_path: str) -> None:
    audio = numpy.array([data, data]).T
    audio = (audio * (2**15 - 1)).astype("<h")
    with wave.open(file_path, "w") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(fs)
        file.writeframes(audio.tobytes())


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
    dsp_plotter.plot(data=mlii, title="MLII", Fs=fs, freqresp=False)
    dsp_plotter.plot(data=v1, title="V1", Fs=fs, freqresp=False)


if __name__ == "__main__":
    main()

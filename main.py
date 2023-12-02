#!/usr/bin/python3
from ecg import Ecg
from dspplotter import DspPlotter


def main() -> None:
    ecg = Ecg()
    fs = ecg.get_fs()
    mlii, v1 = ecg.get_data()
    dsp_plotter = DspPlotter()
    dsp_plotter.plot(data=mlii, title="MLII", Fs=fs, freqresp=False)
    dsp_plotter.plot(data=v1, title="V1", Fs=fs, freqresp=False)


if __name__ == "__main__":
    main()

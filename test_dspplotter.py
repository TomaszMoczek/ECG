import os

from samples import Samples
from dspplotter import DspPlotter

samples = Samples()
dsp_plotter = DspPlotter()

fs = samples.get_fs()
data = samples.get_data()

labels = ("MLII", "V1")


def test_plot():
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot.svg")
    if os.path.exists(file):
        os.remove(file)
    assert os.path.exists(file) == False
    dsp_plotter.plot(
        fs=fs,
        data=data,
        labels=labels,
        div_by_N=True,
        log_freq=True,
        file=file,
    )
    assert os.path.exists(file) == True


def test_spectrogram():
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrogram.svg")
    if os.path.exists(file):
        os.remove(file)
    assert os.path.exists(file) == False
    dsp_plotter.spectrogram(
        fs=fs,
        data=data,
        labels=labels,
        segmentsize=8,
        log_freq=True,
        vmin=-130,
        file=file,
    )
    assert os.path.exists(file) == True

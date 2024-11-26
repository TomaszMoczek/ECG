import os
import numpy

from samples import Samples

samples = Samples()

fs = samples.get_fs()
data = samples.get_data()


def test_get_fs() -> None:
    assert fs == 360


def test_get_data() -> None:
    assert data.shape == (2, 3600) and data.dtype == numpy.float64


def test_write_wave_file_false() -> None:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
    assert (
        Samples.write_wave_file(
            fs=fs,
            data=numpy.empty((3600, 2)),
            file_path=file_path,
        )
        == False
    )


def test_write_wave_file_true() -> None:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
    assert (
        Samples.write_wave_file(
            fs=fs,
            data=data,
            file_path=file_path,
        )
        == True
    )

import os

from samples import Samples

samples = Samples()

fs = samples.get_fs()
data = samples.get_data()


def test_get_fs():
    assert fs == 360


def test_get_data():
    assert len(data[0]) == 3600 and len(data[1]) == 3600


def test_write_wave_file_false():
    mlii = []
    v1 = []
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
    assert Samples.write_wave_file(fs=fs, data=(mlii,), file_path=file_path) == False
    assert Samples.write_wave_file(fs=fs, data=(mlii, v1), file_path=file_path) == False
    mlii.append(1.0)
    assert Samples.write_wave_file(fs=fs, data=(mlii, v1), file_path=file_path) == False
    mlii.append(2.0)
    v1.append(1.0)
    assert Samples.write_wave_file(fs=fs, data=(mlii, v1), file_path=file_path) == False


def test_write_wave_file_true():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav")
    assert Samples.write_wave_file(fs=fs, data=data, file_path=file_path) == True

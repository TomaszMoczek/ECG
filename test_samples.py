from samples import Samples

samples = Samples()


def test_write_wave_file_false():
    assert samples.write_wave_file() == False


def test_get_fs():
    assert samples.get_fs() == 360


def test_get_data():
    mlii, v1 = samples.get_data()
    assert len(mlii) == 3600 and len(v1) == 3600


def test_write_wave_file_true():
    assert samples.write_wave_file() == True

from ecg import Ecg

ecg = Ecg()


def test_get_fs():
    assert ecg.get_fs() == 360


def test_get_data():
    mlii, v1 = ecg.get_data()
    assert len(mlii) == 3600 and len(v1) == 3600

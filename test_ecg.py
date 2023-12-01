from ecg import Ecg


def test_get_data():
    ecg = Ecg()
    mlii, v1 = ecg.get_data("./test.csv")
    assert len(mlii) == 3600 and len(v1) == 3600

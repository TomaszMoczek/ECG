import os


class Ecg:
    def __init__(self) -> None:
        self.fs = 360
        self.file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "samples.csv"
        )

    def get_fs(self) -> int:
        return self.fs

    def get_data(self) -> tuple:
        v1 = []
        mlii = []
        index = -1
        with open(self.file_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            index += 1
            if index <= 1:
                continue
            columns = line.rstrip("\n").split(",")
            mlii.append(float(columns[1]))
            v1.append(float(columns[2]))
        return mlii, v1

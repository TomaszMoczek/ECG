import os
import numpy
import pandas
import wave


class Samples:
    def __init__(self) -> None:
        pass

    def get_fs(self) -> int:
        return 360

    def get_data(self) -> numpy.ndarray:
        file_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "samples.csv"
        )
        data_frame: pandas.DataFrame = pandas.read_csv(file_path, header=[0, 1])
        data: numpy.ndarray = data_frame.to_numpy()
        return data[:, [1, 2]].astype(numpy.float64).T

    def write_wave_file(fs: int, data: numpy.ndarray, file_path: str) -> bool:
        if data.shape != (2, 3600) or data.dtype != numpy.float64:
            return False
        audio = (data.T * (2**15 - 1)).astype(numpy.int16)
        with wave.open(file_path, "w") as file:
            file.setnchannels(2)
            file.setsampwidth(2)
            file.setframerate(fs)
            file.writeframes(audio.tobytes())
        return True

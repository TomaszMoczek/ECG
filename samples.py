import os
import numpy
import wave


class Samples:
    def __init__(self) -> None:
        self.__fs = 360
        self.__mlii = []
        self.__v1 = []

    def get_fs(self) -> int:
        return self.__fs

    def get_data(self) -> tuple:
        index = -1
        self.__mlii.clear()
        self.__v1.clear()
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.csv"), "r"
        ) as file:
            lines = file.readlines()
        for line in lines:
            index += 1
            if index <= 1:
                continue
            items = line.rstrip("\n").split(",")
            self.__mlii.append(float(items[1]))
            self.__v1.append(float(items[2]))
        return self.__mlii, self.__v1

    def write_wave_file(self) -> bool:
        if len(self.__mlii) == 0 and len(self.__v1) == 0:
            return False
        audio = numpy.array([self.__v1, self.__mlii]).T
        audio = (audio * (2**15 - 1)).astype("<h")
        with wave.open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.wav"), "w"
        ) as file:
            file.setnchannels(2)
            file.setsampwidth(2)
            file.setframerate(self.__fs)
            file.writeframes(audio.tobytes())
        return True

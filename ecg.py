class Ecg:
    def __init__(self) -> None:
        self.fs = 360
        self.gain = 200
        self.base = 1024
        pass

    async def get_data(self, file_path: str) -> tuple(list, list):
        mlii = []
        v1 = []
        return mlii, v1

#!/usr/bin/python3
import asyncio
import numpy
from dspplotter import DspPlotter


async def main() -> None:
    dsp_plotter = DspPlotter()
    data = numpy.random.uniform(low=-8.0, high=-3.0, size=3600)
    await dsp_plotter.plot(data=data, title="Test", Fs=360, freqresp=False)


if __name__ == "__main__":
    asyncio.run(main=main())

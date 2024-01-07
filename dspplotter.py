from __future__ import division

import numpy

from scipy import signal
from matplotlib import pyplot


class DspPlotter:
    def __init__(self) -> None:
        pass

    def spectrogram(
        self,
        fs: int,
        data: tuple,
        labels: tuple,
        segmentsize: int = 64,
        overlap: int = 8,
        log_freq: bool = False,
        vmin: int = -160,
        vmax: int = 0,
        file: str = None,
    ) -> None:
        num = len(data)

        if num == 0:
            print(
                "No data is likely provided for the spectrogram to be correctly plotted."
            )
            return

        fig, ax = pyplot.subplots(1, num)
        fig.patch.set_facecolor("#f0f0f0")

        N = len(data[0])
        segments = N // segmentsize - 1
        window = signal.hann(segmentsize * overlap)

        numpy.seterr(all="ignore")

        for i in range(len(data)):
            X = []
            _data = numpy.array(data[i])

            if len(_data) != N:
                print(
                    "Lengths of particular items of data are likely different for the spectrogram to be correctly plotted."
                )
                return

            for j in range(segments - overlap):
                r = range(
                    j * N // segments,
                    j * N // segments + segmentsize * overlap,
                )
                subdata = _data[r]
                subdata = subdata * window
                Y = numpy.fft.fft(subdata)
                Y = Y / len(Y)
                Y = Y[range(len(Y) // 2)]
                Y = numpy.abs(Y)
                Y = 20 * numpy.log10(Y)
                X.append(Y)

            if len(X) == 0:
                print(
                    "Size of the segment provided is likely too big for the spectrogram to be correctly plotted."
                )
                return

            X = numpy.transpose(X)

            _ax = ax[i] if num != 1 else ax

            _ax.set_title(labels[i])
            _ax.set_xlabel("Time (sec)")
            _ax.set_ylabel("Frequency (Hz)")
            _ax.set_xlim([0, N / fs])
            if log_freq:
                _ax.set_yscale("log")
            _ax.set_ylim([1 if log_freq else 0, fs / 2])

            im = _ax.imshow(
                X,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=[0, N / fs, 0, fs / 2],
                interpolation="bicubic",
            )

            pyplot.colorbar(im, ax=_ax)

        pyplot.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

        if file is None:
            pyplot.show()
        else:
            pyplot.savefig(file)

    def plot(
        self,
        fs: int,
        data: tuple,
        labels: tuple,
        freqresp: bool = True,
        padwidth: int = 1024,
        div_by_N: bool = False,
        normalized_freq: bool = False,
        log_freq: bool = False,
        freq_lim: tuple = None,
        freq_dB_lim: tuple = None,
        phaseresp: bool = False,
        phasearg=None,
        file: str = None,
    ) -> None:
        if len(data) == 0:
            print(
                "No data is likely provided for the graph(-s) to be correctly plotted."
            )
            return

        num = 1 + freqresp + phaseresp
        fig, ax = pyplot.subplots(num, 1, figsize=(10, 2.25 * num))
        fig.patch.set_facecolor("#f0f0f0")
        grid_style = {"color": "#777777"}

        N = len(data[0])

        numpy.seterr(all="ignore")

        dataplot = ax[0] if freqresp or phaseresp else ax

        dataplot.set_xlabel("Time (sec)" if div_by_N else "Samples")
        dataplot.set_ylabel("Amplitude (mV)" if div_by_N else "Amplitude")
        dataplot.grid(True, **grid_style)
        dataplot.set_xlim([0, N / fs if div_by_N else N])

        X = (
            numpy.arange(0, N / fs, 1 / fs)
            if div_by_N
            else numpy.linspace(0, N, N, False)
        )

        for i in range(len(data)):
            if len(data[i]) != N:
                print(
                    "Lengths of particular items of data are likely different for the graph(-s) to be correctly plotted."
                )
                return
            dataplot.plot(X, data[i], label=labels[i], linewidth=0.75)

        dataplot.legend(loc="best", shadow=True)

        if freqresp or phaseresp:

            def set_freq(a):
                if normalized_freq:
                    a.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)")
                    if log_freq:
                        a.set_xscale("log")
                        a.set_xlim([0.01, 1])
                    else:
                        a.set_xlim([0, 1])
                    X = numpy.linspace(0, 1, padwidth // 2, False)
                else:
                    a.set_xlabel("Frequency (Hz)")
                    if log_freq:
                        a.set_xscale("log")
                    if freq_lim is not None:
                        a.set_xlim(freq_lim)
                    else:
                        a.set_xlim([1 if log_freq else 0, fs / 2])
                    X = numpy.linspace(0, fs / 2, padwidth // 2, False)
                return X

            padwidth = max(padwidth, N)

            if freqresp:
                freqplot = ax[1]
                freqplot.set_ylabel("Magnitude (dB)")
                if freq_dB_lim is not None:
                    freqplot.set_ylim(freq_dB_lim)
                freqplot.grid(True, **grid_style)
                X = set_freq(freqplot)

            if phaseresp:
                phaseplot = ax[1 + freqresp]
                phaseplot.set_ylabel(
                    r"Phase (${\circ}$)"
                    if phasearg is None
                    else r"Phase shift (${\circ}$)"
                )
                phaseplot.set_ylim([-190, +190])
                phaseplot.grid(True, **grid_style)
                X = set_freq(phaseplot)

            for i in range(len(data)):
                Y = numpy.fft.fft(
                    numpy.pad(
                        data[i], (0, padwidth - N), "constant", constant_values=(0, 0)
                    )
                )
                if div_by_N:
                    Y = Y / padwidth
                Y = Y[range(padwidth // 2)]

                if freqresp:
                    Yfreq = numpy.abs(Y)
                    Yfreq = 20 * numpy.log10(Yfreq)
                    freqplot.plot(X, Yfreq, label=labels[i], linewidth=0.75)

                if phaseresp:
                    if phasearg is not None:
                        if phasearg == "auto":
                            phasearg = (padwidth - 1) * 0.5
                        Y = Y / 1j ** numpy.linspace(
                            0, -(phasearg * 2), len(Y), endpoint=False
                        )
                    Yphase = numpy.angle(Y, deg=True)
                    Yphase = numpy.select([Yphase < -179, True], [Yphase + 360, Yphase])
                    phaseplot.plot(X, Yphase, label=labels[i], linewidth=0.75)

            if freqresp:
                freqplot.legend(loc="best", shadow=True)

            if phaseresp:
                phaseplot.legend(loc="best", shadow=True)

        pyplot.tight_layout(rect=(0, 0.0, 1, 0.98))

        if file is None:
            pyplot.show()
        else:
            pyplot.savefig(file)

from __future__ import division

import numpy

from scipy import signal
from matplotlib import pyplot


class DspPlotter:
    def __init__(self) -> None:
        pass

    def __gen_ticks(self, stop, start=10) -> None:
        yield start
        for s in range(1, 10):
            if start * s > stop:
                yield stop
                return
            yield start * s
        for t in self.__gen_ticks(stop, start * 10):
            yield t

    def __gen_tick_labels(self, stop, start=10) -> None:
        yield (str(int(start)) + "Hz").replace("000Hz", "kHz")
        for s in range(1, 10):
            if start * s > stop:
                yield (str(int(stop)) + "Hz").replace("000Hz", "kHz")
                return
            yield ""
        for t in self.__gen_tick_labels(stop, start * 10):
            yield t

    def spectrogram(
        self,
        data: tuple,
        labels: tuple,
        fs: int,
        horizontal: bool = False,
        segmentsize: int = 64,
        overlap: int = 8,
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

        figsize = (10 if horizontal else 6 * num, 5 * num if horizontal else 6)
        fig, a = (
            pyplot.subplots(num, 1, figsize=figsize)
            if horizontal
            else pyplot.subplots(1, num, figsize=figsize)
        )
        fig.subplots_adjust(top=0.85)
        rect = fig.patch
        rect.set_facecolor("#f0f0f0")

        for i in range(len(data)):
            im = []
            _data = numpy.array(data[i])
            datalen = len(_data)
            segments = datalen // segmentsize - 1
            window = signal.hann(segmentsize * overlap)

            numpy.seterr(all="ignore")

            for segment in range(segments - overlap):
                r = range(
                    segment * datalen // segments,
                    segment * datalen // segments + segmentsize * overlap,
                )
                subdata = _data[r]
                subdata = subdata * window
                n = len(subdata)
                Y = numpy.fft.fft(subdata) / n
                Y = Y[range(len(Y) // 2)]
                Yfreq = 20 * numpy.log10(numpy.abs(Y))
                im.append(Yfreq)

            if len(im) == 0:
                print(
                    "Size of the segment provided is likely too big for the spectrogram to be correctly plotted."
                )
                return

            im = numpy.transpose(im)

            _a = a[i] if num != 1 else a

            _a.set_title(labels[i])
            _a.set_xlabel("Time [sec]")
            _a.set_ylabel("Frequency [Hz]")

            _im = _a.imshow(
                im,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=[0, datalen / fs, 0, fs / 2],
                interpolation="bicubic",
            )

            pyplot.colorbar(_im, ax=_a)

        pyplot.tight_layout(rect=[0, 0.0, 1, 0.94])

        if file is None:
            pyplot.show()
        else:
            pyplot.savefig(file)

    def plot(
        self,
        data: tuple,
        labels: tuple,
        fs: int,
        horizontal: bool = True,
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
        if isinstance(data, (list, tuple, numpy.ndarray)):
            if len(data) == 0:
                print(
                    "No data is likely provided for the graph(-s) to be correctly plotted."
                )
                return

            num = 1 + freqresp + phaseresp
            figsize = (10 if horizontal else 6 * num, 5 * num if horizontal else 6)
            fig, a = (
                pyplot.subplots(num, 1, figsize=figsize)
                if horizontal
                else pyplot.subplots(1, num, figsize=figsize)
            )
            fig.subplots_adjust(top=0.85)
            rect = fig.patch
            rect.set_facecolor("#f0f0f0")
            grid_style = {"color": "#777777"}

            dataplot = a[0] if freqresp or phaseresp else a

            n = len(data[0])

            dataplot.set_xlabel("Time [sec]" if div_by_N else "Samples")
            dataplot.set_ylabel("Amplitude [mV]")
            dataplot.grid(True, **grid_style)
            dataplot.set_autoscalex_on(False)
            dataplot.set_xlim([0, n / fs if div_by_N else n])

            X = (
                numpy.arange(0, n / fs, 1 / fs)
                if div_by_N
                else numpy.linspace(0, n, n, False)
            )

            for i in range(len(data)):
                dataplot.plot(X, data[i], label=labels[i], linewidth=0.75)

            dataplot.legend(loc="best", shadow=True)

            numpy.seterr(all="ignore")

            if freqresp or phaseresp:
                padwidth = max(padwidth, n)
                Y = numpy.fft.fft(
                    numpy.pad(
                        data[0], (0, padwidth - n), "constant", constant_values=(0, 0)
                    )
                )
                if div_by_N:
                    Y = Y / n
                Y = Y[range(padwidth // 2)]

                def set_freq(a):
                    if normalized_freq:
                        a.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)")
                        if log_freq:
                            a.set_xscale("log")
                            a.set_xlim([0.01, 1])
                        else:
                            a.set_xlim([0, 1])
                        X = numpy.linspace(0, 1, len(Y), False)
                    else:
                        a.set_xlabel("Frequency (Hz)")
                        if log_freq:
                            a.set_xscale("log")
                            a.set_xticks(list(self.__gen_ticks(fs / 2)))
                            a.set_xticklabels(list(self.__gen_tick_labels(fs / 2)))
                        if freq_lim is not None:
                            a.set_xlim(freq_lim)
                        else:
                            a.set_xlim([10, fs / 2])
                        X = numpy.linspace(0, fs / 2, len(Y), False)
                    return X

                if freqresp:
                    freqplot = a[1]
                    freqplot.set_ylabel("Amplitude (dB)")
                    if freq_dB_lim is not None:
                        freqplot.set_ylim(freq_dB_lim)
                    freqplot.grid(True, **grid_style)
                    freqplot.set_autoscalex_on(False)
                    X = set_freq(freqplot)
                    Yfreq = 20 * numpy.log10(numpy.abs(Y))
                    freqplot.plot(X, Yfreq, label=labels[0], linewidth=0.75)
                    freqplot.legend(loc="best", shadow=True)

                if phaseresp:
                    phaseplot = a[1 + freqresp]
                    phaseplot.set_ylabel(
                        r"Phase (${\circ}$)"
                        if phasearg is None
                        else r"Phase shift (${\circ}$)"
                    )
                    phaseplot.set_autoscaley_on(False)
                    phaseplot.set_ylim([-190, +190])
                    phaseplot.grid(True, **grid_style)
                    X = set_freq(phaseplot)
                    if phasearg is not None:
                        if phasearg == "auto":
                            phasearg = (n - 1) * 0.5
                        Y = Y / 1j ** numpy.linspace(
                            0, -(phasearg * 2), len(Y), endpoint=False
                        )
                    Yphase = numpy.angle(Y, deg=True)
                    Yphase = numpy.select([Yphase < -179, True], [Yphase + 360, Yphase])
                    phaseplot.plot(X, Yphase, label=labels[0], linewidth=0.75)
                    phaseplot.legend(loc="best", shadow=True)

            pyplot.tight_layout(rect=[0, 0.0, 1, 0.94])

            if file is None:
                pyplot.show()
            else:
                pyplot.savefig(file)

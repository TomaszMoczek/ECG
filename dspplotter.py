from __future__ import division

import numpy
from scipy import signal
import matplotlib
from matplotlib import pyplot
import wave


class DspPlotter:
    def __init__(self) -> None:
        pass

    async def __gen_ticks(self, stop, start=10) -> None:
        yield start
        for s in range(1, 10):
            if start * s > stop:
                yield stop
                return
            yield start * s
        for t in await self.__gen_ticks(stop, start * 10):
            yield t

    async def __gen_tick_labels(self, stop, start=10) -> None:
        yield (str(int(start)) + "Hz").replace("000Hz", "kHz")
        for s in range(1, 10):
            if start * s > stop:
                yield (str(int(stop)) + "Hz").replace("000Hz", "kHz")
                return
            yield ""
        for t in await self.__gen_tick_labels(stop, start * 10):
            yield t

    async def __smooth_colormap(
        self, colors, name="cmap1"
    ) -> matplotlib.colors.LinearSegmentedColormap:
        to_rgb = matplotlib.colors.ColorConverter().to_rgb
        colors = [(p, to_rgb(c)) for p, c in colors]
        result = {"red": [], "green": [], "blue": []}
        for index, item in enumerate(colors):
            pos, color = item
            if pos is not None:
                r, g, b = color
                result["red"].append([pos, r, r])
                result["green"].append([pos, g, g])
                result["blue"].append([pos, b, b])
        cmap = matplotlib.colors.LinearSegmentedColormap(name, result)
        pyplot.register_cmap(name=name, cmap=cmap)
        return cmap

    async def __wavplot(
        self,
        data,
        title="Title",
        file=None,
        segmentsize=512,
        overlap=8,
        Fs=48000,
        vmin=-160,
        vmax=0,
        normalize=False,
    ) -> None:
        cmap = await self.__smooth_colormap(
            [
                (0, "#000000"),
                (1 / 9, "#010325"),
                (2 / 9, "#130246"),
                (3 / 9, "#51026e"),
                (4 / 9, "#9e0379"),
                (5 / 9, "#d6033e"),
                (6 / 9, "#fc4d21"),
                (7 / 9, "#fdc967"),
                (8 / 9, "#f3fab8"),
                (1, "#ffffff"),
            ]
        )

        if not isinstance(data, (list, tuple, numpy.ndarray)):
            w = wave.open(data, "rb")
            Fs = w.getframerate()
            data = (
                numpy.fromstring(w.readframes(w.getnframes()), dtype=numpy.int32)
                / 2147483647.0
            )
        data = numpy.array(data)
        if normalize:
            maxitem = max(abs(numpy.max(data)), abs(numpy.min(data)))
            if maxitem > 0:
                data = data / maxitem

        datalen = len(data)

        async def fast_resample(data, newlen):
            oldlen = len(data)
            result = []
            for i in range(newlen):
                result.append(data[i * oldlen // newlen])
            return numpy.array(result)

        datalen = len(data)
        segments = datalen // segmentsize - 1

        im = []

        window = signal.hann(segmentsize * overlap)

        numpy.seterr(all="ignore")

        for segm in range(segments - overlap):
            r = range(
                segm * datalen // segments,
                segm * datalen // segments + segmentsize * overlap,
            )
            subdata = data[r]
            subdata = subdata * window
            n = len(subdata)
            Y = numpy.fft.fft(subdata) / n
            Y = Y[range(len(Y) // 2)]
            Yfreq = 20 * numpy.log10(numpy.absolute(Y))
            Yfreq = signal.resample(Yfreq, 512)
            Yfreq = numpy.fmax(-300, Yfreq)
            im.append(Yfreq)

        im = numpy.transpose(im)

        pyplot.imshow(
            im,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=[0, datalen / Fs, 0, Fs / 2],
            interpolation="bicubic",
        )
        pyplot.colorbar()

        if not file:
            pyplot.show()
        else:
            pyplot.savefig(file)

    async def plot(
        self,
        data,
        title="Title",
        horizontal=True,
        normalized_freq=False,
        Fs=48000,
        padwidth=1024,
        log_freq=False,
        file=None,
        freqresp=True,
        phaseresp=False,
        dots=False,
        segmentsize=512,
        overlap=8,
        div_by_N=False,
        spectrogram=False,
        vmin=-160,
        vmax=0,
        normalize=False,
        freqticks=[],
        freq_lim=None,
        freq_dB_lim=None,
        phasearg=None,
    ) -> None:
        if phasearg == "auto":
            phasearg = (len(data) - 1) * 0.5

        if isinstance(data, (list, tuple, numpy.ndarray)) and not spectrogram:
            if normalize:
                maxitem = max(abs(numpy.max(data)), abs(numpy.min(data)))
                if maxitem > 0:
                    data = data / maxitem
            n = len(data)
            num = 1 + freqresp + phaseresp
            figsize = (10 if horizontal else 6 * num, 5 * num if horizontal else 6)
            fig, a = (
                pyplot.subplots(num, 1, figsize=figsize)
                if horizontal
                else pyplot.subplots(1, num, figsize=figsize)
            )
            fig.suptitle(title, fontsize=16)
            fig.subplots_adjust(top=0.85)
            rect = fig.patch
            rect.set_facecolor("#f0f0f0")
            style = {"linewidth": 1.4, "color": "#0072bd"}
            grid_style = {"color": "#777777"}

            dataplot = a[0] if freqresp or phaseresp else a

            dataplot.plot(
                numpy.arange(0, n / Fs, 1 / Fs),
                data,
                marker="." if dots else None,
                **style
            )
            dataplot.set_xlabel("Time [s]")
            dataplot.set_ylabel("Amplitude [mV]")
            dataplot.grid(True, **grid_style)
            dataplot.set_autoscalex_on(False)
            dataplot.set_xlim([0, n / Fs])

            numpy.seterr(all="ignore")

            if freqresp or phaseresp:
                padwidth = max(padwidth, n)
                Y = numpy.fft.fft(
                    numpy.pad(
                        data, (0, padwidth - n), "constant", constant_values=(0, 0)
                    )
                )
                Y = Y[range(padwidth // 2)]
                if div_by_N:
                    Y = Y / n
                Yfreq = 20 * numpy.log10(numpy.abs(Y))
                Yfreq = numpy.fmax(-300, Yfreq)

                freq_label = [
                    r"Normalized Frequency ($\times \pi$ rad/sample)",
                    "Frequency (Hz)",
                ]

                async def set_freq(a):
                    if normalized_freq:
                        a.set_xlabel(freq_label[0])
                        X = numpy.linspace(0, 1, len(Y), False)
                        if log_freq:
                            a.set_xscale("log")
                            a.set_xlim([0.01, 1])
                        else:
                            a.set_xlim([0, 1])
                    else:
                        a.set_xlabel(freq_label[1])
                        if log_freq:
                            a.set_xscale("log")

                            a.set_xticks(list(self.__gen_ticks(Fs / 2)))
                            a.set_xticklabels(list(self.__gen_tick_labels(Fs / 2)))
                        X = numpy.linspace(0, Fs / 2, len(Y), False)
                        if freq_lim is not None:
                            a.set_xlim(freq_lim)
                        else:
                            a.set_xlim([10, Fs / 2])
                        a.set_xticks(list(a.get_xticks()) + freqticks)
                    return X

                if freqresp:
                    freqplot = a[1]
                    if freq_dB_lim is not None:
                        freqplot.set_ylim(freq_dB_lim)
                    X = await set_freq(freqplot)
                    freqplot.set_ylabel("Gain (dB)")
                    freqplot.grid(True, **grid_style)
                    freqplot.set_autoscalex_on(False)
                    freqplot.plot(X, Yfreq, **style)

                if phaseresp:
                    phaseplot = a[1 + freqresp]
                    if phasearg is not None:
                        Y = Y / 1j ** numpy.linspace(
                            0, -(phasearg * 2), len(Y), endpoint=False
                        )
                    Yphase = numpy.angle(Y, deg=True)
                    Yphase = numpy.select([Yphase < -179, True], [Yphase + 360, Yphase])
                    X = await set_freq(phaseplot)
                    phaseplot.grid(True, **grid_style)
                    phaseplot.set_ylabel(
                        r"Phase (${\circ}$)"
                        if phasearg is None
                        else r"Phase shift (${\circ}$)"
                    )
                    phaseplot.set_autoscaley_on(False)
                    phaseplot.set_ylim([-190, +190])
                    phaseplot.plot(X, Yphase, **style)

            pyplot.tight_layout(rect=[0, 0.0, 1, 0.94])

            if not file:
                pyplot.show()
            else:
                pyplot.savefig(file)
        else:
            await self.__wavplot(
                data,
                title=title,
                file=file,
                segmentsize=segmentsize,
                overlap=overlap,
                Fs=Fs,
                vmin=vmin,
                vmax=vmax,
                normalize=normalize,
            )

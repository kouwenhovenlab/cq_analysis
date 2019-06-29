"""lmfit model for a sine function"""

import numpy as np
import lmfit


twopi = np.pi * 2


class Model(lmfit.Model):
    """lmfit model for a sine function."""

    @staticmethod
    def func(xs, f, phi, of, scale):
        return scale * np.sin(twopi * f * xs + phi) + of

    def __init__(self):
        super().__init__(self.func)

    def guess(self, data, xs, **kw):
        """Simple initial guess from the FFT of the data."""
        dx = (xs[1:] - xs[:-1]).mean()
        ffty = np.fft.rfft(data)
        ffty[0] = 0.  # remove the DC component (= offset)
        fftx = np.fft.rfftfreq(data.size, d=dx)
        fidx = np.argmax(np.abs(ffty))

        # make the parameters. notes:
        # * since we're fitting with sin (and not cos), we have to rotate the \
        #   fft result to get the right frame.
        # * the noisier the data, the worse this amplitude guess.
        p0 = self.make_params(f=fftx[fidx], phi=np.angle(
            ffty[fidx] * np.exp(1j * np.pi / 2.)),
                              of=data.mean(),
                              scale=(data.max() - data.min()) / 2.)
        return p0

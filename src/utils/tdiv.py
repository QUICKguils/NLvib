import numpy as np

class TimeDivision():
    """TimeDivision -- Handling of time division conversions."""

    def __init__(self):
        self._period = None
        self._circular_frequency = None
        self._frequency = None

    @property
    def T(self):
        return self._period
    @T.setter
    def T(self, period):
        self._period = period
        self._circular_frequency = (2*np.pi) / period
        self._frequency = 1 / period

    @property
    def w(self):
        return self._circular_frequency
    @w.setter
    def w(self, circular_frequency):
        self._period = (2*np.pi) / circular_frequency
        self._circular_frequency = circular_frequency
        self._frequency = circular_frequency / (2*np.pi)

    @property
    def f(self):
        return self._frequency
    @f.setter
    def f(self, frequency):
        self._period = 1 / frequency
        self._circular_frequency = 2*np.pi * frequency
        self._frequency = frequency

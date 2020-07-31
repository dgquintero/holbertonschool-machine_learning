#!/usr/bin/env python3
"""class Normal that represents a Normal distribution"""


class Normal():
    """ Class Normal"""

    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor to call"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = (sum(data) / len(data))
            sigma = 0
            for i in range(len(data)):
                x = (data[i] - self.mean) ** 2
                sigma += x
            self.stddev = (sigma / len(data)) ** (1 / 2)

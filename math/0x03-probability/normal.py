#!/usr/bin/env python3
"""class Normal that represents a Normal distribution"""


class Normal():
    """ Class Normal"""

    e = 2.7182818285
    pi = 3.1415926536

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

    def z_score(self, x):
        """ instance method to calculate the z value"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """ instance method to calculate the z value"""
        return ((self.stddev * z) + self.mean)

    def pdf(self, x):
        """ Method that returns the Probability Density Function"""
        part_1 = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        part_2 = ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))

        return part_1 * Normal.e ** (-part_2)

        
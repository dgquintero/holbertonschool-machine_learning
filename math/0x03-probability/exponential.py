#!/usr/bin/env python3
"""class Exponential that represents a exponential distribution"""


class Exponential():
    """ Class Exponential that calls methos CDF PDF """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """class constructor to call"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = (1 / (sum(data) / len(data)))

    def pdf(self, x):
        """ Method that returns the Probability Density Function"""
        if x < 0:
            return 0
        pdf_p = self.lambtha * Exponential.e**(-(x * self.lambtha))
        return pdf_p

    def cdf(self, x):
        """ Method that returns the Cumulative Distribution Function"""
        if x < 0:
            return 0
        cdf_p = 1 - Exponential.e**(-(x * self.lambtha))
        return pdf_p

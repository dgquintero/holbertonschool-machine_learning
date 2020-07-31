#!/usr/bin/env python3
"""class Binomial that represents a Binomial distribution"""


class Binomial():
    """ Class Binomial that calls methos CDF PDF """

    def __init__(self, data=None, n=1, p=0.5):
        """ Class Binomial that calls methos CDF PDF """
        self.n = int(n)
        self.p = float(p)

        if data is None:
            if self.n <= 0:
                raise ValueError("n must be a postive value")
            elif self.p <= 0 or self.p >= 1:
                raise ValueError("p must be grater tha 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must bea list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            v = 0
            for i in range(len(data)):
                v += ((data[i] - mean) ** 2)
            variance = v / len(data)
            self.p = 1 - (variance / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        pass

    def cdf(self, k):
        pass

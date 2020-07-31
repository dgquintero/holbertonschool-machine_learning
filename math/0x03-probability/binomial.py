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
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / len(data)
                v = 0
                for i in range(len(data)):
                    v += ((data[i] - mean) ** 2)
                variance = v / len(data)
                self.p = 1 - (variance / mean)
                self.n = int(round(mean / self.p))
                self.p = mean / self.n

    def pmf(self, k):
        """ Method that returns the pmf"""
        k = int(k)

        if k > self.n or k < 0:
            return 0
        factor_k = 1
        factor_n = 1
        factor_nk = 1
        for i in range(1, k + 1):
                factor_k *= i
        for i in range(1, self.n + 1):
                factor_n *= i
        for f in range(1, (self.n - k) + 1):
                factor_nk *= f
        comb = factor_n / (factor_nk * factor_k)
        prob = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        pmf = comb * prob
        return pmf

    def cdf(self, k):
        """ Method that returns the Cumulative Distribution Function"""
        k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0
            for i in range(k + 1):
                cdf += self.pmf(i)
            return cdf

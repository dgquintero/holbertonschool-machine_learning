#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    #elif poly == [0]:
        #return [C]
    else:
        integral_poly = [C]
        for i in range(len(poly)):
            x = poly[i] / (i + 1)
            if x.is_integer():
                integral_poly.append(int(x))
            else:
                integral_poly.append(x)
        return integral_poly

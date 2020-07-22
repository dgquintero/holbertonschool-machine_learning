#!/usr/bin/env python3


def poly_derivative(poly):
    if type(poly) == list:
        deriv_poly = [poly[i] * i for i in range(1, len(poly))]
        if deriv_poly != 0:
            return deriv_poly
        return ([0])
    else:
        return None

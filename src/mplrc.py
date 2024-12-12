"""Conventional matplotlib parameters, to remain consistent in all the plots."""

import matplotlib.pyplot as plt


def load_rcparams(style='default') -> None:

    # Use conventional rc parameters for plots
    if style == 'default':
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
        plt.rcParams['figure.figsize'] = (6.34,3.34)
        plt.rcParams['font.size'] = 11
        plt.rcParams['figure.dpi'] = 200

    # Custom rc parameters
    if style == 'custom':
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
        plt.rcParams['font.size'] = 11


if __name__ == '__main__':
    load_rcparams()

# DT2119, Lab 1 Feature Extraction
from scipy import signal
from scipy import fftpack
from lab1_tools import *
import numpy as np

# Function given by the exercise ----------------------------------


def mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    # Audio signals are non-stationary over long periods, but if we examine them over sufficiently short periods, they can be considered stationary.
    # This framing process allows us to analyze the signal in these small, manageable chunks.
    enframed = []
    for i in range(0, len(samples), winshift):
        if i + winlen <= len(samples):
            enframed.append(samples[i:i+winlen])
        else:
            break
    return enframed


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    # Pre-emphasis increases the amplitude of high-frequency components of the signal relative to the lower-frequency components
    # lfilter is a simple first-order Finite Impulse Response (FIR) filter
    emphasis = signal.lfilter([1, -p], [1], input)  # y[n] = x[n] − p⋅x[n−1]
    return emphasis


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    # A window function is to taper the beginning and end of a signal.
    # When analyzing signals, especially in the context of Fourier transforms, we often work with a finite section of a signal.
    # However, taking a finite section of an infinite or longer signal and treating it as if it were the whole signal introduces discontinuities at the edges of this section.
    # These discontinuities can lead to artifacts in the analysis, such as spectral leakage.
    # The purpose of applying a window function is to minimize these discontinuities by smoothly reducing the amplitude of the signal to zero at the edges of the section being analyzed.
    window = signal.hamming(len(input[0]), sym=False)
    return window * input


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    # The power spectrum is calculated from the FFT, which represents the power of each frequency component present in the frame.

    # The sampling frequency should be greater than twice the highest frequency in the signal
    # Given that example['samplingrate'] = 20000 -> f_max = 10000, frequencies above this threshold are subject to aliasing
    ff = fftpack.fft(input, nfft)
    return ff.real ** 2 + ff.imag ** 2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    # The filterbank is designed to mimic the human ear's response more closely than the linearly spaced frequency bands used in the original FFT.
    # The Mel scale relates perceived frequency of a pure tone to its actual measured frequency.
    # The Mel scale defines a sound signal at 1000 Hz that is 40 decibels above the human hearing threshold as the reference point of 1000 mel.
    # When the frequency is above 500Hz, as the frequency increases, every time the human ear feels the same amount of pitch change, the required frequency change becomes larger and larger.
    # This results in the four octaves above 500 Hz on the Hertz scale (one octave being twice the frequency) corresponding to only two octaves on the Mel scale.
    # Filters are spaced linearly at low frequencies and logarithmically at high frequencies, which more accurately represents the human ear's resolution.

    # The filters in a Mel filterbank are triangular and are linearly spaced on the Mel scale, which translates to a logarithmic spacing on the linear frequency scale.
    trf = trfbank(samplingrate, len(input[0]))
    Mspec = np.dot(input, trf.T)
    # After passing through the Mel filterbank, the logarithm of each filter's energy is taken.
    # This step is performed because human hearing perceives sound intensity logarithmically.
    return np.log(Mspec)


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    # The DCT helps to de-correlate the filterbank coefficients and produce a compressed representation of the filterbank energies.
    # The coefficients that result from the DCT are the Mel Frequency Cepstral Coefficients.
    # Typically, only the first 12-13 of these coefficients are used as features in voice recognition tasks, as they contain the most significant information.
    ceps = fftpack.dct(input)[:, 0:nceps]
    return ceps


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    # Dynamic Time Warping (DTW) is used for measuring similarity between two temporal sequences which may vary in speed.
    N = len(x)
    M = len(y)
    LD = dist

    # Compute accumulated distances
    # Initialize accumulated distance matrix with infinity
    AD = np.full((N, M), np.inf)
    AD[0, 0] = LD[0, 0]
    for i in range(1, N):
        AD[i, 0] = LD[i, 0] + AD[i-1, 0]
    for j in range(1, M):
        AD[0, j] = LD[0, j] + AD[0, j-1]

    for i in range(1, N):
        for j in range(1, M):
            cost = LD[i, j]
            AD[i, j] = cost + min(AD[i - 1, j], AD[i, j - 1], AD[i - 1, j - 1])

    # Compute the global distance normalized by the sum of the lengths of the two sequences
    d = AD[-1, -1] / (N + M)

    # Backtrack to find path
    path = [(N-1, M-1)]
    i, j = N-1, M-1
    while i > 0 or j > 0:
        # Check previous steps to determine the minimum
        steps = [(i-1, j), (i, j-1), (i-1, j-1)]
        costs = [AD[step] if step[0] >= 0 and step[1]
                 >= 0 else np.inf for step in steps]
        min_cost_index = np.argmin(costs)
        i, j = steps[min_cost_index]
        path.append((i, j))

    path.reverse()  # Reverse path to start from the beginning

    return d, LD, AD, path

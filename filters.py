import analyse
import numpy as np


def filter_horizontal_freq(orig):

    trimmedData1, trimmedData2 = analyse.cut_me(orig )
    merged_data = np.vstack((trimmedData1, trimmedData2))

    # first FFT
    FFT_spectrum = analyse.FFT(merged_data)
    FFT_spectrum_filtered = analyse.filter_frequency(FFT_spectrum, 0, 1, 'horizontal')
    filtered_cutted = analyse.reverse_FFT(FFT_spectrum_filtered)

    disturbance_image = merged_data - filtered_cutted
    filtered_image = analyse.subtract_disturbance(orig, disturbance_image)
    analyse.gimme_noise(orig, filtered_image)

    return filtered_image

def filter_horizontal_freq_bis(orig):

    FFT_spectrum = analyse.FFT(orig)
    FFT_spectrum_filtered = analyse.filter_frequency(FFT_spectrum, 0, 1, 'horizontal')
    filtered_cutted = analyse.reverse_FFT(FFT_spectrum_filtered)

    analyse.gimme_noise(orig, filtered_cutted)

    return filtered_cutted

def filter_dark_spots(orig):

    FFT_spectrum = analyse.FFT(orig)
    dark = analyse.filter_dark_spots(FFT_spectrum, orig)
    filtered_cutted = analyse.reverse_FFT(dark)

    analyse.gimme_noise(orig, filtered_cutted)

    return filtered_cutted

def filter_frequency(orig, which='low'):

    FFT_spectrum = analyse.FFT(orig)
    if which == 'low':
        filtered_spectrum=analyse.gaussian_lowpass(FFT_spectrum, 60)
    else:
        filtered_spectrum = analyse.gaussian_highpass(FFT_spectrum, 1)
    filtered = analyse.reverse_FFT(filtered_spectrum)
    

    return filtered

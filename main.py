import argparse
import analyse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Get some data and images")
    parser.add_argument("-f", "--file", required=True, help="Path to the .mdma file")
    parser.add_argument("-p", "--parameter", default="imagMreal",
                        choices=["real", "imag", "magnitude", "phase", "phaseMreal", "imagMreal"],
                        help="Select the parameter to use: real_data, imag_data, magnitude, phase, phase filtered by real, imag filtered by real. Default is imagMreal (imag filtered by real).")

    args = parser.parse_args()

    real_data, imag_data, magnitude, phase, phaseMreal, imagMreal  = analyse.go_get_them(args.file)

    # Based on the -p argument, select the corresponding data
    data_dict = {
        "real": real_data,
        "imag": imag_data,
        "magnitude": magnitude,
        "phase": phase,
        "phaseMreal": phaseMreal,
        "imagMreal": imagMreal
    }
    orig = data_dict[args.parameter]


    trimmedData1, trimmedData2 = analyse.cut_me(orig)

    merged_data = np.vstack((trimmedData1, trimmedData2))

    wooshFF = analyse.FFT(merged_data)
    horiz = analyse.filter_frequency(wooshFF, 0, 1, 'horizontal')
    firstImage = analyse.reverse_FFT(horiz)

    disturbance_image = merged_data - firstImage
    corrected_image = analyse.subtract_disturbance(orig, disturbance_image)
    analyse.compute_percentage(orig, corrected_image)
    analyse.gimme_noise(orig, corrected_image)


    # szum termiczny
    # dark = analyse.filter_dark_spots(corrected_image, orig)
    #
    # disturbance_image = disturbance_image - dark
    # corrected_image = analyse.subtract_disturbance(corrected_image, disturbance_image)
    #
    # analyse.compute_percentage(imag_data, corrected_image)
    # analyse.gimme_noise(imag_data, corrected_image)



if __name__ == "__main__":
    main()

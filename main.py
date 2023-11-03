#!/usr/bin/env python3
import argparse
import analyse
import filters
# matplotlib.use('Agg')  # Ustaw back-end na Agg



def main():
    parser = argparse.ArgumentParser(description="Get some data and images")
    parser.add_argument("-f", "--file", required=True, help="Path to the .mdma file")
    parser.add_argument("-p", "--parameter", default="imagMreal",
                        choices=["real", "imag", "magnitude", "phase", "phaseMreal", "imagMreal"],
                        help="Select the parameter to use: real_data, imag_data, magnitude, phase, phase filtered by real, imag filtered by real. Default is imagMreal (imag filtered by real).")

    args = parser.parse_args()

    real_data, imag_data, magnitude, phase, phaseMreal, imagMreal, orig_phase, orig_magnitude, orig_real, orig_imag = analyse.go_get_them(args.file)

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
    filtered_H_freq = filters.filter_horizontal_freq(orig)
    filtered_low_freq = filters.filter_frequency(filtered_H_freq, 'low')



if __name__ == "__main__":
    main()

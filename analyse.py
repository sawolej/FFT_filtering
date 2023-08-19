

import openpyxl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def go_get_them(file_path):

    df = pd.read_csv(file_path, header=None)

    range_values = df.iloc[0,1:].values

    real_images = []
    imag_images = []

    for index, row in df.iloc[1:].iterrows():
        parameter_value = row[0]
        complex_values = row[1:]

        real_values = np.real(complex_values.astype(complex))
        imag_values = np.imag(complex_values.astype(complex))

        real_images.append(real_values)
        imag_images.append(imag_values)

    real_images = np.array(real_images)
    imag_images = np.array(imag_images)

    imag_images = np.nan_to_num(imag_images)
    real_images = np.nan_to_num(real_images)

    real_data = (real_images - np.min(real_images)) / (np.max(real_images) - np.min(real_images))
    imag_data = (imag_images - np.min(imag_images)) / (np.max(imag_images) - np.min(imag_images))

    magnitude = np.sqrt(np.square(real_data) + np.square(imag_data))
    phase = np.arctan2(imag_data, real_data)

    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))

    phaseMreal = phase - real_data
    imagMreal = imag_data - real_data

    phaseMreal = (phaseMreal - np.min(phaseMreal)) / (np.max(phaseMreal) - np.min(phaseMreal))
    imagMreal = (imagMreal - np.min(imagMreal)) / (np.max(imagMreal) - np.min(imagMreal))
    # phase = phase[:20, :]

    output_folder = "cyferki"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save phase data
    phase_df = pd.DataFrame(phase)  # Limit to 20 rows as per your comment
    phase_df.to_csv(os.path.join(output_folder, 'phase_data.csv'), index=False)
    phase_df.to_excel(os.path.join(output_folder, 'phase_data.xlsx'), index=False, engine='openpyxl')

    # Save magnitude data
    magnitude_df = pd.DataFrame(magnitude)
    magnitude_df.to_csv(os.path.join(output_folder, 'magnitude_data.csv'), index=False)
    magnitude_df.to_excel(os.path.join(output_folder, 'magnitude_data.xlsx'), index=False, engine='openpyxl')

    # Save imaginary part data
    imag_df = pd.DataFrame(imag_data)
    imag_df.to_csv(os.path.join(output_folder, 'imag_data.csv'), index=False)
    imag_df.to_excel(os.path.join(output_folder, 'imag_data.xlsx'), index=False, engine='openpyxl')

    # Save real part data
    real_df = pd.DataFrame(real_data)
    real_df.to_csv(os.path.join(output_folder, 'real_data.csv'), index=False)
    real_df.to_excel(os.path.join(output_folder, 'real_data.xlsx'), index=False, engine='openpyxl')




    # Create the subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    img1 = ax1.imshow(real_data, cmap='binary_r')
    ax1.set_title('Real Part')
    ax1.invert_yaxis()
    fig.colorbar(img1, ax=ax1)

    img2 = ax2.imshow(imag_data, cmap='binary_r')
    ax2.set_title('Imaginary Part')
    ax2.invert_yaxis()
    fig.colorbar(img2, ax=ax2)

    img3 = ax3.imshow(magnitude, cmap='binary_r')
    ax3.set_title('Magnitude')
    ax3.invert_yaxis()
    fig.colorbar(img3, ax=ax3)

    img4 = ax4.imshow(phase, cmap='binary_r')
    ax4.set_title('Phase')
    ax4.invert_yaxis()
    fig.colorbar(img4, ax=ax4)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('1.png')
    plt.close()

    return real_data, imag_data, magnitude, phase, phaseMreal, imagMreal

def FFT(data):
    F2D = np.fft.fft2(data)
    F2D_przesuniecie = np.fft.fftshift(F2D)  # Przesuwa zero częstotliwości do centrum

    # plt.imshow(np.log(np.abs(F2D_przesuniecie)), cmap='gray')
    # plt.colorbar()
    # plt.title('Spektrum amplitudy')
    # plt.savefig('FFT.png')
    # plt.savefig('spectrumFFT.png')
    # plt.show()

    return F2D_przesuniecie


def filter_frequency(F2D_shifted, d, r_block, direction='horizontal'):
    print(F2D_shifted.shape)
    h, w = F2D_shifted.shape
    mask = np.ones((h, w), np.uint8)
    centrum_y, centrum_x = h // 2, w // 2

    if direction == 'horizontal':
        y_up = int(h // 2 - d - r_block)
        y_down = int(h // 2 + d)
        mask[y_up:y_up + 2 * r_block, :] = 0

    else:  # vertical
        x_left = int(w // 2 - d - r_block)
        x_right = int(w // 2 + d)
        mask[:, x_left:x_left + 2 * r_block] = 0

    mask[centrum_y, centrum_x] = 1
    # plt.imshow(np.log(np.abs(F2D_shifted * mask)), cmap='gray')
    # plt.colorbar()
    # plt.title('filtered')
    # plt.savefig('filteredPhase.png')
    # # plt.show()
    # plt.close()
    return F2D_shifted * mask

def reverse_FFT(filtered_F2D_shifted):
    filtered_F2D = np.fft.ifftshift(filtered_F2D_shifted)

    reconstructed_image = np.fft.ifft2(filtered_F2D)

    reconstructed_image_real = np.real(reconstructed_image)


    # plt.imshow(reconstructed_image_real, cmap='gray')
    # plt.colorbar()
    # plt.title('filtered reverse FFT')
    # plt.savefig('reverseFiltered.png')
    # # plt.show()
    # plt.close()

    return reconstructed_image_real

def get_horizontal_freq(F2D_przesuniecie):
    h, w = F2D_przesuniecie.shape

    centrum_y, centrum_x = h // 2, w // 2

    górny_pas_y = np.argmax(np.abs(F2D_przesuniecie[centrum_y - 10:centrum_y, centrum_x])) + centrum_y - 10
    dolny_pas_y = np.argmax(np.abs(F2D_przesuniecie[centrum_y + 1:centrum_y + 10, centrum_x])) + centrum_y + 1

    odleglosc_górny = centrum_y - górny_pas_y
    odleglosc_dolny = dolny_pas_y - centrum_y

    czestotliwosc_górny = -odleglosc_górny / h
    czestotliwosc_dolny = odleglosc_dolny / h

    print(f"Częstotliwość górnego pasa: {czestotliwosc_górny:.4f} cykli na piksel")
    print(f"Częstotliwość dolnego pasa: {czestotliwosc_dolny:.4f} cykli na piksel")

    return czestotliwosc_górny, czestotliwosc_dolny

def get_vertical_freq(F2D_przesuniecie):
    h, w = F2D_przesuniecie.shape
    centrum_y, centrum_x = h // 2, w // 2

    lewy_pas_x = np.argmax(np.abs(F2D_przesuniecie[centrum_y, centrum_x - 10:centrum_x])) + centrum_x - 10
    prawy_pas_x = np.argmax(np.abs(F2D_przesuniecie[centrum_y, centrum_x + 1:centrum_x + 10])) + centrum_x + 1

    odleglosc_lewy = centrum_x - lewy_pas_x
    odleglosc_prawy = prawy_pas_x - centrum_x

    czestotliwosc_lewy = -odleglosc_lewy / w
    czestotliwosc_prawy = odleglosc_prawy / w

    print(f"Częstotliwość lewego pasa: {czestotliwosc_lewy:.4f} cykli na piksel")
    print(f"Częstotliwość prawego pasa: {czestotliwosc_prawy:.4f} cykli na piksel")

    return czestotliwosc_lewy, czestotliwosc_prawy


def find_pixels_below_threshold(F2D_shifted, percentage=100):
    threshold = np.max(np.abs(F2D_shifted)) * percentage / 100.0
    max_value = np.max(np.abs(F2D_shifted))
    min_value = np.min(np.abs(F2D_shifted))

    print("Max value:", max_value)
    print("Min value:", min_value)
    coords = np.where(np.abs(F2D_shifted) < threshold)


    for coord in coords:
        F2D_shifted[coord[0], coord[1]] = 0
    print("do i even work")
    # plt.imshow(np.log(np.abs(F2D_shifted)), cmap='gray')
    # plt.colorbar()
    # plt.title('dots w func1')
    # plt.show()

    return list(zip(coords[0], coords[1]))

def filter_pixels_by_coordinates(data, coordinates):

    filtered_data = np.copy(data)
    h, w = data.shape
    centrum_y, centrum_x = h // 2, w // 2
    for coord in coordinates:
        if (coord[0] == centrum_y and coord[1] == centrum_x):
            continue  # Pomiń środkowy piksel
        filtered_data[coord[0], coord[1]] = 0
    print("do i even work")
    # plt.imshow(np.log(np.abs(filtered_data)), cmap='gray')
    # plt.colorbar()
    # plt.title('dots dots')
    # plt.show()


    return filtered_data

def filter_dark_spots(F2D_shifted, orig):
    magnitude_spectrum = np.abs(F2D_shifted)
    log_magnitude_spectrum = np.log(magnitude_spectrum + 1)  # dodajemy 1, aby uniknąć log(0)

    threshold = np.max(log_magnitude_spectrum) / 10
    dark_pixels = log_magnitude_spectrum< threshold
    F2D_shifted[dark_pixels] = 0

    # filtered_data = np.fft.ifft2(np.fft.ifftshift(F2D_shifted))

    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(log_magnitude_spectrum, cmap='gray')
    # plt.title('Oryginalne Spektrum Amplitudy')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.log(np.abs(F2D_shifted) + 1), cmap='gray')
    # plt.title('Przefiltrowane Spektrum Amplitudy')
    #
    # plt.savefig('darksFilterFFTSpectrum.png')
    # plt.show()
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.abs(filtered_data), cmap='gray')
    # plt.title('filter orig')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(orig, cmap='gray')
    # plt.title('orig')
    # plt.savefig('darksFilterFFT.png')
    # plt.show()

    return F2D_shifted
def low_filter(data):
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = 60  # promień filtra
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1

    return data * mask


def high_filter(data, cutoff_frequency=0):

    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    r = cutoff_frequency  # promień filtra
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    fshift = data * mask


    return fshift


def remove_diagonal_lines(F2D_shifted):

    rows, cols = F2D_shifted.shape

    center_i, center_j = rows // 2, cols // 2

    diagonal_length = np.sqrt(rows ** 2 + cols ** 2) * 0.25

    start = center_i - int(diagonal_length / np.sqrt(2))
    end = center_i + int(diagonal_length / np.sqrt(2))

    def condition_for_diagonal(i, j):
        if i == center_i and j == center_j:
            return False

        margin = 1
        is_in_diagonal_range = start <= i <= end and start <= j <= end

        if is_in_diagonal_range and (abs(i - j) <= margin or abs(i + j - rows) <= margin):
            return True

        return False

    mask = np.ones_like(F2D_shifted)
    for i in range(rows):
        for j in range(cols):
            if condition_for_diagonal(i, j):
                mask[i, j] = 0

    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.log(np.abs(F2D_shifted) + 1), cmap='gray')
    # plt.title('Oryginalne Spektrum Amplitudy')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.log(np.abs(F2D_shifted * mask) + 1), cmap='gray')
    # plt.title('Przefiltrowane Spektrum Amplitudy')
    # plt.savefig('noiseSpectrum.png')
    # plt.show()

    return F2D_shifted * mask

def gimme_noise(orig, filtered):

    nois = orig - filtered

    #save

    plt.imshow(orig, cmap='gray')
    plt.colorbar()
    plt.title('original')
    plt.show()

    plt.imshow(filtered, cmap='gray')
    plt.colorbar()
    plt.title('filtered')
    plt.show()


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(orig, cmap='gray')
    plt.title('Oryginalne dane')

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(filtered), cmap='gray')

    plt.title('Przefiltrowane dane')

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(nois), cmap='gray')
    plt.title('szum')
    plt.savefig('noises.png')
    plt.show()
    signal_mean = np.std(orig)
    noise_std = np.std(filtered)



    if noise_std == 0:
        raise ValueError("Noise standard deviation is 0 lol")

    snr = 20 * np.log10(signal_mean / noise_std)
    print("SNR: ", snr)

    return nois


def compute_percentage(signal, noise):
    signal_energy = np.sum(np.abs(signal) ** 2)
    noise_energy = np.sum(np.abs(noise) ** 2)
    print("%%%: ",(noise_energy / signal_energy) * 100)
    return (noise_energy / signal_energy) * 100

def subtract_disturbance(original, disturbance):

    repeat_factor = int(np.ceil(original.shape[0] / disturbance.shape[0]))

    tiled_disturbance = np.tile(disturbance, (repeat_factor, 1))

    tiled_disturbance = tiled_disturbance[:original.shape[0], :]

    subtracted = original - tiled_disturbance

    return subtracted


def cut_me(data):
    """
    POC: correction using naive separation of areas with samples with lines cut
    """
    data_uint8 = (data * 255).astype(np.uint8)

    _, binary = cv2.threshold(data_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(binary, 100, 200, apertureSize=3)

    edge_points_per_row = np.sum(edges != 0, axis=1)

    # Sorting the edge lengths in descending order
    sorted_indices = np.argsort(edge_points_per_row)[::-1]

    # Finding two largest edge lengths that are separated by at least 10 pixels
    for i in range(len(sorted_indices) - 1):
        if abs(sorted_indices[i] - sorted_indices[i + 1]) >= 10:
            two_largest_indices = [sorted_indices[i], sorted_indices[i + 1]]
            break

    for idx in two_largest_indices:
        cv2.line(data_uint8, (0, idx), (data_uint8.shape[1] - 1, idx), (0, 0, 255), 2)

    # Uncomment below for visualizing results
    plt.imshow(edges, cmap='binary_r')
    plt.colorbar()
    plt.title("edg")
    plt.show()
    plt.imshow(data_uint8, cmap='binary_r')
    plt.colorbar()
    plt.title("edges")
    plt.show()

    rows, _ = data.shape
    up = min(two_largest_indices)
    dp = rows - max(two_largest_indices)

    return cut_me_harder(up, dp, data)


def cut_me_harder(up, dp, data):
    rows, _ = data.shape
    print(rows)
    print (up, dp)
    trimmed_dataUP = data[:up, :]
    trimmed_dataDP = data[rows - dp:, :]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(trimmed_dataUP, cmap='gray')
    plt.title('Oryginalne Spektrum Amplitudy')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(trimmed_dataDP, cmap='gray')
    plt.title('Przefiltrowane Spektrum Amplitudy')

    plt.savefig('darksFilterFFTSpectrum.png')
    plt.show()

    if up > dp:
        return trimmed_dataUP, trimmed_dataDP
    else:
        return trimmed_dataDP, trimmed_dataUP


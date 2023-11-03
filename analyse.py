import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_and_save(images, titles, save_path):
    """
    Wyświetla obrazy w formie wykresu i zapisuje je jako SVG.

    Args:
        images (list): Lista obrazów do wyświetlenia.
        titles (list): Tytuły dla obrazów.
        save_path (str): Ścieżka, gdzie zapisany zostanie wykres w formacie SVG.
    """

    # Ustal rozmiar wykresu
    plt.figure(figsize=(20, 5))
    plt.rcParams["axes.titlesize"] = 16


    # Iteruj przez obrazy i tytuły
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout(pad=2.0)

    plt.savefig(save_path, format='svg', transparent=True)
    plt.show()


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
    orig_real = real_data
    orig_imag = imag_data
    real_data = real_data - imag_data
    imag_data = imag_data - real_data
    imag_data = (imag_data - np.min(imag_data)) / (np.max(imag_data) - np.min(imag_data))
    real_data = (real_data - np.min(real_data)) / (np.max(real_data) - np.min(real_data))


    imagMreal = imag_data - real_data


    magnitude = np.sqrt(np.square(real_data) + np.square(imag_data))
    phase = np.arctan2(imag_data, real_data)

    orig_magnitude = np.sqrt(np.square(orig_real) + np.square(orig_imag))
    orig_phase = np.arctan2(orig_imag, orig_real)

    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
    phaseMreal = phase - real_data

    phaseMreal = (phaseMreal - np.min(phaseMreal)) / (np.max(phaseMreal) - np.min(phaseMreal))
    imagMreal = (imagMreal - np.min(imagMreal)) / (np.max(imagMreal) - np.min(imagMreal))

    output_folder_num = "cyferki"
    if not os.path.exists(output_folder_num):
        os.makedirs(output_folder_num)

    output_folder = "pics"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save phase data
    phase_df = pd.DataFrame(phase)  # Limit to 20 rows as per your comment
    phase_df.to_csv(os.path.join(output_folder_num, 'phase_data.csv'), index=False)
    phase_df.to_excel(os.path.join(output_folder_num, 'phase_data.xlsx'), index=False, engine='openpyxl')

    # Save magnitude data
    magnitude_df = pd.DataFrame(magnitude)
    magnitude_df.to_csv(os.path.join(output_folder_num, 'magnitude_data.csv'), index=False)
    magnitude_df.to_excel(os.path.join(output_folder_num, 'magnitude_data.xlsx'), index=False, engine='openpyxl')

    # Save imaginary part data
    imag_df = pd.DataFrame(imag_data)
    imag_df.to_csv(os.path.join(output_folder_num, 'imag_data.csv'), index=False)
    imag_df.to_excel(os.path.join(output_folder_num, 'imag_data.xlsx'), index=False, engine='openpyxl')

    # Save real part data
    real_df = pd.DataFrame(real_data)
    real_df.to_csv(os.path.join(output_folder_num, 'real_data.csv'), index=False)
    real_df.to_excel(os.path.join(output_folder_num, 'real_data.xlsx'), index=False, engine='openpyxl')




    # # Create the subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    img1 = ax1.imshow(real_data, cmap='binary_r')
    ax1.set_title(r'Real Part: ($\mathit{a\prime=a-b}$)')
    ax1.invert_yaxis()
    fig.colorbar(img1, ax=ax1)

    img2 = ax2.imshow(imag_data, cmap='binary_r')
    ax2.set_title(r'Imaginary Part: ($\mathit{b\prime=b-a}$)')
    ax2.invert_yaxis()
    fig.colorbar(img2, ax=ax2)

    img3 = ax3.imshow(magnitude, cmap='binary_r')
    ax3.set_title(r'Magnitude ($\mathit{\sqrt{a\prime^2 + b\prime^2}}$)')

    ax3.invert_yaxis()
    fig.colorbar(img3, ax=ax3)

    img4 = ax4.imshow(phase, cmap='binary_r')
    ax4.set_title(r"Phase ($\mathit{arctan\left(\frac{b\prime}{a\prime}\right)}$)")


    ax4.invert_yaxis()
    fig.colorbar(img4, ax=ax4)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(output_folder, '1god.png'), transparent=True)  # Zmiana na svg i zapis w folderze "pics"
    plt.close()



    # Create the subplots

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    img1 = ax1.imshow(orig_real, cmap='binary_r')
    ax1.set_title(r'Real Part: ($\mathit{a}$)')
    ax1.invert_yaxis()
    fig.colorbar(img1, ax=ax1)

    img2 = ax2.imshow(orig_imag, cmap='binary_r')
    ax2.set_title(r'Imaginary Part: ($\mathit{b}$)')
    ax2.invert_yaxis()
    fig.colorbar(img2, ax=ax2)

    img3 = ax3.imshow(orig_magnitude, cmap='binary_r')
    ax3.set_title(r'Magnitude ($\mathit{\sqrt{a^2 + b^2}}$)')

    ax3.invert_yaxis()
    fig.colorbar(img3, ax=ax3)

    img4 = ax4.imshow(orig_phase, cmap='binary_r')
    ax4.set_title(r"Phase ($\mathit{arctan\left(\frac{b}{a}\right)}$)")
    ax4.invert_yaxis()
    fig.colorbar(img4, ax=ax4)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(output_folder, '1bad.png'), transparent=True)
    plt.close()

    return real_data, imag_data, magnitude, phase, phaseMreal, imagMreal, orig_phase, orig_magnitude, orig_real, orig_imag

def FFT(data):
    F2D = np.fft.fft2(data)
    F2D_przesuniecie = np.fft.fftshift(F2D)

    plt.imshow(np.log(np.abs(F2D_przesuniecie)), cmap='gray')
    plt.colorbar()
    plt.title('Spektrum amplitudy')
    #plt.savefig(os.path.join("pics", 'FFT.png'))
    #plt.show()
    plt.close()

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
    plt.imshow(np.log(np.abs(F2D_shifted * mask)), cmap='gray')
    plt.colorbar()
    plt.title('filtered spectrum')
    #plt.savefig(os.path.join("pics", 'filtered_spectrum.png'))
    # plt.show()
    plt.close()
    return F2D_shifted * mask

def reverse_FFT(filtered_F2D_shifted):
    filtered_F2D = np.fft.ifftshift(filtered_F2D_shifted)

    reconstructed_image = np.fft.ifft2(filtered_F2D)

    reconstructed_image_real = np.real(reconstructed_image)

    plt.imshow(reconstructed_image_real, cmap='gray')
    plt.colorbar()
    plt.title('filtered reverse FFT')
    plt.savefig(os.path.join("pics", 'filtered.png'))
    #plt.show()
    plt.close()

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

    plt.imshow(np.log(np.abs(F2D_shifted)), cmap='gray')
    plt.colorbar()
    plt.title('dots w func1')
    #plt.savefig(os.path.join("pics", 'dotsFunc1.png'))
    #plt.show()
    plt.close()

    return list(zip(coords[0], coords[1]))

def filter_pixels_by_coordinates(data, coordinates):

    filtered_data = np.copy(data)
    h, w = data.shape
    centrum_y, centrum_x = h // 2, w // 2
    for coord in coordinates:
        if (coord[0] == centrum_y and coord[1] == centrum_x):
            continue  # Pomiń środkowy piksel
        filtered_data[coord[0], coord[1]] = 0
    plt.imshow(np.log(np.abs(filtered_data)), cmap='gray')
    plt.colorbar()
    plt.title('dots dots')
    #plt.savefig(os.path.join("pics", 'dotsDots.png'))
    #plt.show()
    plt.close()


    return filtered_data

def filter_dark_spots(F2D_shifted, orig):
    magnitude_spectrum = np.abs(F2D_shifted)
    log_magnitude_spectrum = np.log(magnitude_spectrum + 1)  # + 1 aby uniknąć log(0)
    x = 1
    median = np.median(log_magnitude_spectrum)
    std = np.std(log_magnitude_spectrum)
    threshold = median*0.1 + (x * std)

    dark_pixels = log_magnitude_spectrum < threshold
    F2D_shifted[dark_pixels] = 0

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(log_magnitude_spectrum, cmap='gray')
    plt.title('Oryginalne Spektrum Amplitudy')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(F2D_shifted) + 1), cmap='gray')
    plt.title('Przefiltrowane Spektrum Amplitudy')
    #plt.show()
    plt.close()

    plt.savefig(os.path.join("pics", 'darksFilterFFTSpectrum.png'))

    return F2D_shifted


def gaussian_lowpass(data, cutoff_frequency= None):
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2

    # auto ustawienie wartości cutoff_frequency
    if cutoff_frequency is None:
        cutoff_frequency = min(rows, cols) / 2

    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    x, y = np.meshgrid(x, y)

    # maska gaussowska
    mask = np.exp(-((x ** 2 + y ** 2) / (2 * (cutoff_frequency ** 2))))

    fshift = data * mask

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Maska Gaussowska')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(fshift) + 1), cmap='gray')
    plt.title('Dane po zastosowaniu maski Gaussowskiej')

    #plt.show()
    plt.close()

    return fshift


def gaussian_highpass(data, cutoff_frequency=None):
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2

    # auto ustawienie wartości cutoff_frequency
    if cutoff_frequency is None:
        cutoff_frequency = min(rows, cols) / 2

    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    x, y = np.meshgrid(x, y)

    # maska gaussowska
    mask = 1 - np.exp(-((x ** 2 + y ** 2) / (2 * (cutoff_frequency ** 2))))

    fshift = data * mask

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Maska Gaussowska')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(fshift) + 1), cmap='gray')
    plt.title('Dane po zastosowaniu maski Gaussowskiej')

    #plt.show()
    plt.close()

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
    compute_percentage(orig, filtered)
    #save

    plt.imshow(orig, cmap='gray')
    plt.colorbar()
    plt.title('original')
    #plt.show()

    plt.imshow(filtered, cmap='gray')
    plt.colorbar()
    plt.title('filtered')
    # plt.show()


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(orig, cmap='gray')
    plt.title('Oryginalne dane')

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(filtered), cmap='gray')

    plt.title('Przefiltrowane dane')

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(nois), cmap='gray')
    plt.title('Szum')
    plt.savefig(os.path.join('pics', 'noises.png'), transparent=True)
    #plt.show()
    plt.close()

    signal_mean = np.std(orig)
    noise_std = np.std(filtered)



    if noise_std == 0:
        raise ValueError("Noise standard deviation is 0 why")

    snr = 20 * np.log10(signal_mean / noise_std)
    print("SNR: ", snr)

    return nois


def compute_percentage(signal, orig):
    noise = orig - signal
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
    assert data_uint8.dtype == np.uint8, "Data type is not uint8"

    _, binary = cv2.threshold(data_uint8, 200, 210, cv2.THRESH_BINARY)

    # Closing operation
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(closing, 180, 200, apertureSize=7)

    edge_points_per_row = np.sum(edges != 0, axis=1)

    # Sorting the edge lengths in descending order
    sorted_indices = np.argsort(edge_points_per_row)[::-1]

    # Finding two largest edge lengths that are separated by at least 10 pixels
    two_largest_indices = None
    for i in range(len(sorted_indices) - 1):
        if abs(sorted_indices[i] - sorted_indices[i + 1]) >= 10:
            two_largest_indices = [sorted_indices[i], sorted_indices[i + 1]]
            break

    if two_largest_indices is None:
        print("could not find two separate indices!")
        return None  # todo: other handling

    # copy of the data to draw lines on
    data_with_lines = data_uint8.copy()
    for idx in two_largest_indices:
        cv2.line(data_with_lines, (0, idx), (data_with_lines.shape[1] - 1, idx), 255, 2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(data_uint8, cmap='gray')
    axs[0].set_title('Original Data')

    axs[1].imshow(edges, cmap='binary_r')
    axs[1].set_title('debug info')

    axs[2].imshow(data_with_lines, cmap='gray')
    axs[2].set_title('Data with Lines')
    plt.savefig(os.path.join("pics", 'objectisolate.png'))

    #plt.show()

    rows, _ = data.shape
    up = min(two_largest_indices) - 1
    dp = rows - max(two_largest_indices) - 1

    return cut_me_harder(up, dp, data)


def adapt_filter(data):
    """
    POC: correction using naive separation of areas with samples with lines cut
    """
    data_uint8 = (data * 255).astype(np.uint8)

    _, binary = cv2.threshold(data_uint8, 150, 255, cv2.THRESH_BINARY)
    # _, binary = cv2.threshold(data_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # closing operation
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(closing, 100, 200, apertureSize=3)

    # normalize
    object_mask = closing > 0
    object_mask = ~object_mask

    normalized = cv2.normalize(data_uint8, None, 200, 220, cv2.NORM_MINMAX)
    normalized_object = np.where(object_mask, normalized, 0)

    # reconstruct the image with normalized object and original background
    result = np.where(object_mask, normalized_object, data_uint8)
    # kernel = np.ones((9, 9), np.uint8)
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(data_uint8, cmap='gray')
    plt.title('Original Data')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')

    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='gray')
    plt.title('Filtered Data')

    #plt.show()

    return result

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

    #plt.savefig('darksFilterFFTSpectrum.png')
    #plt.show()
    plt.close()


    if up > dp:
        return trimmed_dataUP, trimmed_dataDP
    else:
        return trimmed_dataDP, trimmed_dataUP



def nanmedian(window):
    """Funkcja do obliczania mediany ignorując wartości NaN"""
    return np.nanmedian(window)

# def thereshold_background_correction(phaseMreal):
#     """
#     POC: auto - correction using thereshold. just horrible approach for datasets like with huge noises overlaps
#     the samples. Better than lines cut for datasets when we dont see the whole probe or dont have much background.
#     still may be need for manually adjust parameters based of type of object or streight of the cable noises
#     """
#
#     # thresholding
#     thresh = threshold_otsu(phaseMreal)
#     object_mask = phaseMreal > thresh + 0.1
#
#     phase_masked = phaseMreal.copy()
#
#     phase_masked[object_mask] = np.nan
#
#     y_non_nan, x_non_nan = np.nonzero(~np.isnan(phase_masked))
#
#     min_x, max_x = np.min(x_non_nan), np.max(x_non_nan)
#     min_y, max_y = np.min(y_non_nan), np.max(y_non_nan)
#
#     square_mask = np.zeros_like(phase_masked, dtype=bool)
#     square_mask[min_y:max_y+1, min_x:max_x+1] = True
#
#     border_mask = square_mask & np.isnan(phase_masked)
#
#     phase_masked[border_mask] = np.nanmean(phase_masked[~object_mask])
#
#
#     y_center, x_center = (min_y + max_y) // 2, (min_x + max_x) // 2  # środek kwadratu
#     y, x = np.ogrid[-y_center:phase_masked.shape[0]-y_center, -x_center:phase_masked.shape[1]-x_center]
#     circle_mask = x*x + y*y <= 2*2  # maska dla koła o średnicy 4 piksele
#     # phase_masked[circle_mask & (phase_masked > 0.4)] = np.nan
#     nan_coords = np.where(circle_mask & (phase_masked > 0.3))
#     circle_mask = x * x + y * y <= 2 * 2  # maska dla koła o średnicy 4 piksele
#     object_pixels_mask = ~np.isnan(phase_masked)
#     original_circle_values = phase_masked.copy()
#     original_circle_values[~circle_mask] = np.nan
#
#     phase_masked = generic_filter(phase_masked, nanmedian, size=30)
#     phase_masked[~object_pixels_mask] = np.nan  # Przywracanie wartości NaN dla pikseli, które nie są częścią obiektu
#
#     phase_masked[circle_mask] = original_circle_values[circle_mask]
#
#     phase_masked = np.nan_to_num(phase_masked, nan=1.0)
#
#     phase_masked = (phase_masked - np.min(phase_masked)) / (np.max(phase_masked) - np.min(phase_masked))
#     phase_masked[nan_coords] = 0.3
#     plt.imshow(phase_masked, cmap='binary_r')
#     plt.colorbar()
#     plt.title('phase_masked')
#     plt.show()
#
#     return phase_masked


def visualize_data(data1, data2, title1=" ", title2=" ", filename1='1.png', filename2='2.png'):
    """
    3D and 2D visualisation
    """
    normalized_amplitude = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))

    normalized_phase = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))

    phase_color = plt.cm.hsv(normalized_phase)

    phase_color[..., 3] = normalized_amplitude

    plt.imshow(phase_color)
    plt.title(title1)
    plt.savefig(filename1)
    plt.close()

    x = np.arange(data1.shape[1])
    y = np.arange(data1.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ax.plot_surface(x, y, data2, color='b', alpha=0.5)

    ax.plot_surface(x, y, data1, color='r', alpha=0.5)

    plt.title(title2)
    plt.savefig(filename2)
    plt.show()
    plt.close()



def experimantal_cut(data):
    data_uint8 = (data * 255).astype(np.uint8)
    assert data_uint8.dtype == np.uint8, "Data type is not uint8"

    _, binary = cv2.threshold(data_uint8, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


    edges = cv2.Canny(closing, 100, 200, apertureSize=7)
    orig_edges = edges
    dilate_kernel = np.ones((3, 3), np.uint8)  # 5x5 - 2 pikesele
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)

    # identify the edges that touch the image boundary
    height, width = edges.shape
    left_most = np.where(edges[:, 0] > 0)
    right_most = np.where(edges[:, -1] > 0)
    top_most = np.where(edges[0, :] > 0)
    bottom_most = np.where(edges[-1, :] > 0)

    if left_most[0].size > 0:
        y1 = left_most[0][0]
        height = edges.shape[0]

        if y1 > height / 2:
            y2 = 0  # górny róg obrazu
        else:
            y2 = height - 1  # dolny róg obrazu

        x = 0
        cv2.line(edges, (x, y1), (x, y2), 255, 1)

    if right_most[0].size > 0:
        y1 = right_most[0][0]
        height = edges.shape[0]

        if y1 > height / 2:
            y2 = 0  # górny róg obrazu
        else:
            y2 = height - 1  # dolny róg obrazu

        x = width - 1
        cv2.line(edges, (x, y1), (x, y2), 255, 1)

    if top_most[0].size > 0:
        x1 = top_most[0][0]
        width = edges.shape[1]

        if x1 > width / 2:
            x2 = 0  # lewy róg obrazu
        else:
            x2 = width - 1  # prawy róg obrazu

        y = 0
        cv2.line(edges, (x1, y), (x2, y), 255, 1)

    if bottom_most[0].size > 0:
        print('bottom')
        print(bottom_most[0][-1])
        print(height-3)
        x1 = bottom_most[0][0]
        width = edges.shape[1]

        if x1 > width / 2:
            x2 = 0  # lewy róg obrazu
        else:
            x2 = width - 1  # prawy róg obrazu

        y = height - 1
        cv2.line(edges, (x1, y), (x2, y), 255, 1)


    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create a mask with the contour filled
    mask = np.zeros_like(data_uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=-1)  # fill the detected contours

    # find the top and bottom rows of the mask
    rows = np.where(mask.sum(axis=1) > 0)[0]

    top_row, bottom_row = rows[0], rows[-1]
    object_height = bottom_row - top_row + 1
    print('top_row, bottom_row')
    print(top_row, bottom_row)

    # copy the rows from below or from above
    if data_uint8.shape[0] - bottom_row - 1 >= object_height:
        replacement_data = data_uint8[bottom_row + 1: bottom_row + 1 + object_height, :]
    else:
        if top_row >= object_height:
            replacement_data = data_uint8[top_row - object_height: top_row, :]
        else:
            single_row_data = data_uint8[top_row, :]
            replacement_data = np.tile(single_row_data, (object_height, 1))

    # replace the area inside the contour
    result = data_uint8.copy()
    j=top_row
    for i in range(top_row, top_row + object_height):
        if i + object_height < data_uint8.shape[0]:  # if wychodzi poza obiekt
            row_data = data_uint8[i + object_height, :]
        else:
            print("yeah")
            row_data = data_uint8[(j + object_height) % data_uint8.shape[0], :]
            j = (j + 1) % data_uint8.shape[0]
            if (j==32):
                j = top_row

        # whwere end of the object
        mask_row = mask[i, :]
        object_start = np.where(mask_row > 0)[0][0]
        object_end = np.where(mask_row > 0)[0][-1]

        result[i, object_start:object_end + 1] = row_data[object_start:object_end + 1]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(data_uint8, cv2.COLOR_BGR2RGB))
    plt.title('Dane oryginalne', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(orig_edges, cmap='gray')
    plt.title('Detekcja próbki', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Usunięcie próbki', fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('pics', 'cuttmme.pdf'), transparent=True)

    #plt.show()
    plt.close()

    return result / 255.0, orig_edges, result

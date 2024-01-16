import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import math

def generate_qpsk_signal(binary_data):
    symbol_mapping = {
        '00': np.pi,       # 180°
        '01': np.pi/2,     # 90°
        '10': 3*np.pi/2,   # 270°
        '11': 0,           # 0°
    }

    carrier_freq = 1  
    samples_per_symbol = 100 
    total_symbols = len(binary_data) // 2
    x = np.linspace(0, 2 * np.pi * total_symbols, total_symbols * samples_per_symbol)

    qpsk_signal = np.array([])

    for i in range(0, len(binary_data), 2):
        symbol = binary_data[i:i+2]
        phase = symbol_mapping[symbol]
        wave_segment = np.sin(carrier_freq * x[:samples_per_symbol] + phase)
        qpsk_signal = np.concatenate([qpsk_signal, wave_segment])
        x = x[samples_per_symbol:]

    scaled_qpsk_signal = 2 * qpsk_signal + 3

    sample_n = 20
    sampling_interval = 100//sample_n
    sampled_indices = np.arange(0, len(scaled_qpsk_signal), sampling_interval)
    sampled_qpsk_signal = scaled_qpsk_signal[sampled_indices]

    sampled_qpsk_signal_rounded = np.round(sampled_qpsk_signal).astype(int)

    return sampled_qpsk_signal_rounded





def decode_to_ascii(rank_outputs_W, sample_n, window_size=3, sigma=1):
    def gaussian_filter(data, sigma):
        return gaussian_filter1d(data, sigma)

    def sin_wave(x, amplitude, phase, offset):
        return amplitude * np.sin(x + phase) + offset

    data = rank_outputs_W

    subarrays = [data[i:i+sample_n] for i in range(0, len(data), sample_n)]
    smoothed_data_gaussian = [gaussian_filter(array, sigma) for array in subarrays]

    resampled_subarrays = []
    for original_array, smoothed_array in zip(subarrays, smoothed_data_gaussian):
        if len(smoothed_array) < len(original_array):
            extended_smoothed_array = np.pad(smoothed_array, (0, len(original_array) - len(smoothed_array)), 'edge')
            resampled_subarrays.append(extended_smoothed_array)
        else:
            resampled_subarrays.append(smoothed_array[:len(original_array)])

    x = np.linspace(0, 2 * np.pi, sample_n)
    initial_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    amplitude_phase_list = []

    for array in resampled_subarrays:
        best_fit_phase = None
        smallest_error = float('inf')
        params = [0, 0, 0]

        for initial_phase in initial_phases:
            try:
                params, _ = curve_fit(sin_wave, x, array, p0=[2, initial_phase, 3])
                fitted_curve = sin_wave(x, *params)
                error = np.sum((fitted_curve - array) ** 2)
                if error < smallest_error:
                    best_fit_phase = params[1]
                    smallest_error = error
            except RuntimeError:
                continue

        amplitude_phase_list.append((params[0], params[1]))

    process_data = []
    for amplitude, phase in amplitude_phase_list:
        if amplitude < 0:
            process_data.append(phase / np.pi - 1)
        else:
            process_data.append(phase / np.pi)

    decode_bin = ''
    for j in process_data:
        if 0.25 < j <= 0.75:
            decode_bin += '01'
        elif 0.75 < j <= 1.25:
            decode_bin += '00'
        elif 1.25 < j <= 1.75:
            decode_bin += '10'
        elif j > 1.75 or 0 < j <= 0.25:
            decode_bin += '11'
            
    print(decode_bin)
    
    ascii_character = chr(int(decode_bin, 2))
    return ascii_character

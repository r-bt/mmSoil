import numpy as np

max_freq = 77 * 10**9
c = 3 * 10**8  # speed of light in m/s

e_prime_max = 16.2
e_prime_min = 13.1

e_prime_prime_max = 7.2
e_prime_prime_min = 5.8

def calc_attenuation(e_prime, e_prime_prime, freq):
    """
    Calculate the attenuation (in dB/m) for given dielectric properties and frequency.
    e_prime: real part of dielectric constant
    e_prime_prime: imaginary part of dielectric constant
    freq: frequency in Hz
    """
    omega = 2 * np.pi * freq
    k = omega / c
    alpha = k * np.sqrt(0.5 * (np.sqrt(e_prime**2 + e_prime_prime**2) - e_prime))
    return alpha

def calculate_skin_depth(e_prime, e_prime_prime, freq):
    """
    Calculate the skin depth (in meters) for given dielectric properties and frequency.
    e_prime: real part of dielectric constant
    e_prime_prime: imaginary part of dielectric constant
    freq: frequency in Hz
    """
    alpha = calc_attenuation(e_prime, e_prime_prime, freq)
    return 1/alpha

if __name__ == "__main__":
    skin_depth_max = calculate_skin_depth(e_prime_min, e_prime_prime_min, max_freq)
    skin_depth_min = calculate_skin_depth(e_prime_max, e_prime_prime_max, max_freq)

    print(f"Maximum Skin Depth (m): {skin_depth_max:.4f}")
    print(f"Minimum Skin Depth (m): {skin_depth_min:.4f}")

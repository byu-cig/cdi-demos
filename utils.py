import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd


def fft(arr):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))


def ifft(arr):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr)))


def log(comp_arr):
    amp = np.abs(comp_arr)
    phi = np.angle(comp_arr)
    return np.log(amp+1) * np.exp(1j*phi)


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def pad_to_size(arr, N_new):
    # Assuming square array
    N = arr.shape[0]
    pad = (N_new - N) // 2
    if 2*pad + N == N_new:
        return np.pad(arr, ((pad, pad), (pad, pad)))
    elif 2*pad + N == N_new - 1:
        return np.pad(arr, ((pad, pad+1), (pad, pad+1)))
    else:
        raise IndexError(f'Error padding to desired size: N_new={N_new}, N_old={N}, N_pad={pad}, N_out={2*pad + N}')


def comp_to_rgb(comp_img):
    amp = normalize(np.abs(comp_img))
    phi = normalize(np.angle(comp_img))
    one = np.ones_like(amp)
    hsv = np.dstack((phi, one, amp))
    return colors.hsv_to_rgb(hsv)


def save_image(img, cmap='plasma'):
    root = tk.Tk()
    root.withdraw()
    fname = fd.asksaveasfilename(defaultextension='png', filetypes=[('PNG', 'png')])
    plt.imsave(fname, img, cmap=cmap)

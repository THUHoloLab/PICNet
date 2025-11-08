import os
import numpy as np
import torch
import math
import scipy.io
from math import pi
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.functional import pad

from function.functions import center_crop


# ------------------------------------------------------------
# FFT helpers
# ------------------------------------------------------------
def torch_fft(x: torch.Tensor) -> torch.Tensor:
    """
    2D FFT with fftshift on the last two dims.

    Args:
        x: complex tensor, shape (B, C, H, W)

    Returns:
        torch.Tensor: shifted FFT result
    """
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))


def torch_ifft(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse 2D FFT with ifftshift on the last two dims.

    Args:
        x: complex tensor, shape (B, C, H, W)

    Returns:
        torch.Tensor: inverse FFT result
    """
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)))


# ------------------------------------------------------------
# Angular Spectrum Method (ASM) with Zernike aberration
# ------------------------------------------------------------
def ASM(
    O: torch.Tensor,
    aber: torch.Tensor,
    lamb: float,
    px: float,
    aperture: float,
    mag: float,
    device: torch.device,
    zero_padding: bool,
) -> torch.Tensor:
    """
    Angular Spectrum propagation with pupil + Zernike aberration.

    Args:
        O: complex object field, shape (B, 1, H, W)
        aber: aberration coefficients per sample, shape (B, 12) (assumed)
        lamb: wavelength (m)
        px: pixel size (m)
        aperture: numerical aperture
        mag: magnification
        device: torch device
        zero_padding: whether to pad object field before propagation

    Returns:
        torch.Tensor: propagated complex field, cropped to original size
    """
    batch, c, Sh, Sw = O.shape

    # --------------------------------------------------------
    # Spatial frequency coordinates (fx, fy)
    # --------------------------------------------------------
    if zero_padding:
        # replicate-pad to double size
        O = pad(O, pad=(Sh // 2, Sh // 2, Sw // 2, Sw // 2), mode="replicate")
        fx = np.arange(Sh * 2) / 2 - Sh // 2
        fy = np.arange(Sw * 2) / 2 - Sw // 2
    else:
        fx = np.arange(Sh) - Sh // 2
        fy = np.arange(Sw) - Sw // 2

    # normalized spatial frequencies
    fx = fx / (Sh * px)
    fy = fy / (Sw * px)

    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    kxm, kym = np.meshgrid(kx, ky)

    # wave number
    k0 = 2 * pi / lamb
    NA = aperture
    M = mag

    # pupil cutoff frequency
    cutoff_frequency = k0 * NA / M
    cutoff_frequency = np.float64(cutoff_frequency)

    # pupil mask
    term = (kxm ** 2 + kym ** 2) <= cutoff_frequency ** 2
    pupil = torch.from_numpy(
        np.repeat(term.reshape((1, 1, len(fx), len(fy))), batch, axis=0)
    ).to(device=device)

    # --------------------------------------------------------
    # Zernike aberration
    # --------------------------------------------------------
    mat_data = scipy.io.loadmat("zernike_poly.mat")
    zernike_poly = mat_data["zernike_poly"]
    zernike_basis = torch.from_numpy(
        zernike_poly.reshape(len(fy), len(fy), 15)
    ).to(device=device)

    fn_zernike = torch.zeros((batch, 1, len(fx), len(fy))).to(device=device)

    for i in range(aber.shape[0]):

        fn_zernike[i, 0, :, :] = (
            aber[i, 0] * zernike_basis[:, :, 3]
            + aber[i, 1] * zernike_basis[:, :, 4]
            + aber[i, 2] * zernike_basis[:, :, 5]
            + aber[i, 3] * zernike_basis[:, :, 6]
            + aber[i, 4] * zernike_basis[:, :, 7]
            + aber[i, 5] * zernike_basis[:, :, 8]
            + aber[i, 6] * zernike_basis[:, :, 9]
            + aber[i, 7] * zernike_basis[:, :, 10]
            + aber[i, 8] * zernike_basis[:, :, 11]
            + aber[i, 9] * zernike_basis[:, :, 12]
            + aber[i, 10] * zernike_basis[:, :, 13]
            + aber[i, 11] * zernike_basis[:, :, 14]
        )

    # define aberrated coherent transfer function
    aberrated_CTF = torch.exp(1j * fn_zernike) * pupil

    # --------------------------------------------------------
    # Propagation process
    # --------------------------------------------------------
    O_fft = torch_fft(O)
    get_y = torch_ifft(aberrated_CTF * O_fft)

    # crop back to original size
    crop_y = center_crop(get_y, Sh)

    return crop_y


# ------------------------------------------------------------
# Physical forward model module
# ------------------------------------------------------------
class Physical_Forward_Model(nn.Module):

    def __init__(self, args):
        super(Physical_Forward_Model, self).__init__()
        self.wavelength = args.wavelength
        self.pixel_size = args.pixel_size
        self.phase_normalize = args.phase_normalize
        self.aperture = args.aperture
        self.mag = args.magnification
        self.device = args.device
        self.zero_padding = True

    def forward(self, phase: torch.Tensor, aber: torch.Tensor) -> torch.Tensor:

        amplitude = torch.ones_like(phase)
        phase_scaled = phase * self.phase_normalize
        obj_field = amplitude * torch.exp(1j * phase_scaled)

        holo_field = ASM(
            obj_field,
            aber,
            self.wavelength,
            self.pixel_size,
            self.aperture,
            self.mag,
            self.device,
            self.zero_padding,
        )

        intensity = torch.pow(torch.abs(holo_field), 2)
        I_norm = torch.zeros_like(intensity)
        for i in range(intensity.size(0)):
            ch = intensity[i, 0, :, :]
            min_val = ch.min()
            max_val = ch.max()
            I_norm[i, 0, :, :] = (ch - min_val) / (max_val - min_val + 1e-8)

            # debug check
            if torch.isnan(I_norm[i, 0, :, :]).any():
                print(f"I_norm channel {i} contains NaN!")

        return I_norm

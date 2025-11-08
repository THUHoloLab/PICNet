import torch
from torch import nn

# ------------------------------------------------------------
# Feature-domain loss
# ------------------------------------------------------------
class Loss_feature(nn.Module):
    def __init__(self, weight: float = 1) -> None:

        super().__init__()
        self.weight = weight

    def forward(self, prediction, measurement):

        batch_size, c, h, w = prediction.size()

        gradx_pred = torch.cat([prediction[:, :, :, 1:] - prediction[:, :, :, :-1], prediction[:, :, :, :1] - prediction[:, :, :, -1:]], dim=3)
        grady_pred = torch.cat([prediction[:, :, 1:, :] - prediction[:, :, :-1, :], prediction[:, :, :1, :] - prediction[:, :, -1:, :]], dim=2)

        gradx_meas = torch.cat([measurement[:, :, :, 1:] - measurement[:, :, :, :-1], measurement[:, :, :, :1] - measurement[:, :, :, -1:]], dim=3)
        grady_meas = torch.cat([measurement[:, :, 1:, :] - measurement[:, :, :-1, :], measurement[:, :, :1, :] - measurement[:, :, -1:, :]], dim=2)

        loss_fn = torch.abs(gradx_pred - gradx_meas) + torch.abs(grady_pred - grady_meas)
        loss_sum = loss_fn.sum()
        return self.weight * loss_sum / (batch_size * c * h * w)


# ------------------------------------------------------------
# Adversarial Loss

# Reference: J. Seo et al., “Deep-learning-driven end-to-end metalens imaging,” Adv. Photon. 6(6) 066002 (2024)
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

# ------------------------------------------------------------
def Adversarial_loss_D(real_logits, fake_logits, weight=0.5):
    """
     Adversarial loss for discriminator

    Args:
        real_logits: D(x) for real samples
        fake_logits: D(G(z)) for generated samples
        weight: optional scaling factor (default 0.5)

    Returns:
        torch.Tensor: scalar loss for discriminator
    """
    loss_real = torch.mean(nn.ReLU(inplace=True)(1.0 - real_logits))
    loss_fake = torch.mean(nn.ReLU(inplace=True)(1.0 + fake_logits))
    return weight * (loss_real + loss_fake)


def Adversarial_loss_G(fake_logits, weight=0.5):
    """
    Adversarial loss for generator

    Args:
        fake_logits: D(G(z)) for generated samples
        weight: optional scaling factor (default 0.5)

    Returns:
        torch.Tensor: scalar loss for generator
    """
    return -weight * torch.mean(fake_logits)



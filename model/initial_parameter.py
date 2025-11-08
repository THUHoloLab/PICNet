import argparse
from math import pi

def parse_args():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------
    # Dataset parameters
    # ------------------------------------------------------------
    parser.add_argument("--train_phase_ratio", type=float, default=1.0,
                        help="Fraction of phase data used for training (0~1)")
    parser.add_argument("--train_intensity_ratio", type=float, default=1.0,
                        help="Fraction of intensity data used for training (0~1)")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="Crop size for input and label images")

    # ------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--iterations", type=int, default=300000,
                        help="Total training iterations")
    parser.add_argument("--chk_iter", type=int, default=1500,
                        help="Interval for printing training loss")
    parser.add_argument("--model_chk_iter", type=int, default=1500,
                        help="Interval for testing model")

    # Learning rate
    parser.add_argument("--lr_disc", type=float, default=1e-5,
                        help="Learning rate for discriminator")
    parser.add_argument("--lr_gen", type=float, default=2e-4,
                        help="Learning rate for generator")
    parser.add_argument("--lr_decay_epoch", type=int, default=50,
                        help="Number of epochs before each learning rate decay")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5,
                        help="Decay rate for learning rate scheduler")

    # Loss coefficients
    parser.add_argument("--adv_coefficient", type=float, default=1.0,
                        help="Weight for adversarial loss")
    parser.add_argument("--pha_coefficient", type=float, default=60.0,
                        help="Weight for phase L1 loss")
    parser.add_argument("--feature_pha_coefficient", type=float, default=100.0,
                        help="Weight for phase feature-domain loss")
    parser.add_argument("--aberration_coefficient", type=float, default=40.0,
                        help="Weight for aberration loss")
    parser.add_argument("--intensity_coefficient", type=float, default=50.0,
                        help="Weight for intensity loss")
    parser.add_argument("--phase_normalize", type=float, default=pi,
                        help="Normalization factor for phase")

    # ------------------------------------------------------------
    # System parameters
    # ------------------------------------------------------------
    parser.add_argument("--wavelength", type=float, default=625e-9,
                        help="Wavelength")
    parser.add_argument("--pixel_size", type=float, default=5.86e-6,
                        help="Pixel size on the sensor")
    parser.add_argument("--aperture", type=float, default=0.28,
                        help="Numerical aperture of objective")
    parser.add_argument("--magnification", type=float, default=10.0,
                        help="Magnification factor")

    args = parser.parse_args()
    return args

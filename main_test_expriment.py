
#==============================================================================
# This script provides the experimental evaluation of the Physics-Informed Cycle-Consistent Network
# (PICNet) for adaptive aberration correction in quantitative phase microscopy.
#
# Author:   Danlin Xu
# Date:     2025/10/30
#==============================================================================

import time
import scipy.io as sio
from torchvision import transforms
from model.initial_parameter import parse_args
from model.network_architecture import Phase_Generator, Aberration_Generator, Discriminator
from model.physical_model import Physical_Forward_Model
from function.data_saving import *
from function.functions import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
set_seed(0)

if __name__ == '__main__':

    # load parameters
    args = parse_args()

    # load model weight
    params = torch.load('D:\\xdl\\Figure_element\\Fig5\\PGGAN\\model_pth_z6_v2\\396.pth')

    # load experimental_data_path
    kk = 1
    dataset_path = "D:\\xdl\\PICNet\\dataset\\expdata\\quantitative_phase_target"
    dataset_path = os.path.join(dataset_path, f'{kk}.mat')
    result_path = "D:\\xdl\\PICNet\\dataset\\result_exp"

    # =================================================
    # load experimental measured intensity
    # =================================================

    intensity = sio.loadmat(dataset_path)['holo']
    transform = transforms.ToTensor()
    measured_intensity = transform(intensity)
    measured_intensity = measured_intensity.unsqueeze(0)

    # =================================================
    # define network
    # =================================================

    # define physical forward model
    Propagator = Physical_Forward_Model(args).to(device=args.device)  # physical model

    # define phase generator for reconstructing sample phase
    Phase_G = Phase_Generator(num_in=1, num_out=1).to(device=args.device)

    # define aberration generator for reconstructing pupil aberrations
    Aberration_G = Aberration_Generator(out_class=12).to(device=args.device)

    # define the discriminator
    Phase_D = Discriminator().to(device=args.device)  # discriminator

    # =================================================
    # load model weights
    # =================================================
    Phase_G.load_state_dict(params['Phase_G_state_dict'])
    Aberration_G.load_state_dict(params['Aberration_G_state_dict'])

    Phase_G.eval()
    Aberration_G.eval()

    # =================================================
    # Reconstructing phase and pupil aberrations
    # =================================================

    measured_intensity = center_crop(measured_intensity, args.crop_size).float()
    start_time = time.perf_counter()
    retrieved_phase = Phase_G(measured_intensity)
    retrieved_aber_coefficients = Aberration_G(measured_intensity)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # =================================================
    # Show result
    # =================================================

    measured_intensity_numpy = measured_intensity.detach().numpy()[0][0]
    plt.imshow(measured_intensity_numpy, cmap='gray')
    plt.colorbar()
    plt.show()

    retrieved_phase_numpy = retrieved_phase.detach().numpy()[0][0]
    retrieved_phase_numpy = retrieved_phase_numpy * args.phase_normalize
    plt.imshow(retrieved_phase_numpy, cmap='hot', vmin=np.min(retrieved_phase_numpy), vmax=np.max(retrieved_phase_numpy))
    plt.colorbar()
    plt.show()

#==============================================================================
# This script provides the test implementation of the Physics-Informed Cycle-Consistent Network
# (PICNet) for adaptive aberration correction in quantitative phase microscopy.
#
# Author:   Danlin Xu
# Date:     2025/10/30
#==============================================================================

from torch.utils.data import DataLoader
from torchvision import transforms
from model.initial_parameter import parse_args
from model.data_reading import QPM_Dataset
from model.network_architecture import Phase_Generator, Aberration_Generator, Discriminator
from model.physical_model import Physical_Forward_Model
from function.loss_function import *
from function.data_saving import *
from function.metric_function import *
from function.functions import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# matplotlib.use('Agg')
set_seed(0)

if __name__ == '__main__':

    # load parameters
    args = parse_args()

    # load model weight
    params = torch.load('D:\\xdl\\Figure_element\\Fig5\\PGGAN\\model_pth_z6_v2\\396.pth')

    # environment setting
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available")
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define data path
    dataset_path = "D:\\xdl\\PICNet\\dataset\\testdata"
    saving_path = "D:\\xdl\\PICNet\\result_recon"
    make_path(saving_path)

    # =================================================
    # load test dataset
    # =================================================

    # load test data of measured intensity
    test_intensity = QPM_Dataset(
        root=dataset_path,
        data_type=["holo"],
        image_set="testdata",
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test_intensity_loader = DataLoader(test_intensity, batch_size=1, shuffle=False)

    # load training data of ground-truth phase
    test_phase = QPM_Dataset(
        root=dataset_path,
        data_type=["pha"],
        image_set="testdata",
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_phase_loader = DataLoader(test_phase, batch_size=1, shuffle=False)
    N_val = test_intensity.__len__()

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

    # =================================================
    # define loss
    # =================================================
    criterion_cycle = nn.L1Loss().to(device=args.device)
    criterion_feature = Loss_feature().to(device=args.device)

    Phase_G.eval()
    Aberration_G.eval()

    # load test dataset
    test_gth_phase = iter(DataLoader(test_phase, batch_size=1, shuffle=False))
    test_measured_intensity = iter(DataLoader(test_intensity, batch_size=1, shuffle=False))

    sum_psnr_pha, sum_ssim_pha, sum_rmse_pha, sum_pcc_pha = 0, 0, 0, 0

    for nn in range(N_val):
        test_gth_pha = next(test_gth_phase)
        test_mea_inten, test_gth_aber = next(test_measured_intensity)
        test_gth_aber = test_gth_aber.squeeze(0).squeeze(1)

        test_gth_pha = center_crop(test_gth_pha, args.crop_size).to(args.device).float() / (args.phase_normalize)
        test_mea_inten = center_crop(test_mea_inten, args.crop_size).to(args.device).float()
        test_gth_aber = test_gth_aber.to(args.device).float()

        # recover phase
        test_retrieved_pha = Phase_G(test_mea_inten)

        # recover aberration coefficients
        test_retrieved_aber = Aberration_G(test_mea_inten)

        # test loss
        testloss_pha = criterion_cycle(test_retrieved_pha, test_gth_pha)
        testloss_feature_pha = criterion_feature(test_retrieved_pha, test_gth_pha)
        testloss_aber = criterion_cycle(test_retrieved_aber, test_gth_aber)

        # from tensor to numpy
        test_mea_inten = test_mea_inten.cpu().detach().numpy()[0][0]
        test_gth_pha = test_gth_pha.cpu().detach().numpy()[0][0]
        test_retrieved_pha = test_retrieved_pha.cpu().detach().numpy()[0][0]
        test_gth_aber = test_gth_aber.cpu().detach().numpy()
        test_retrieved_aber = test_retrieved_aber.cpu().detach().numpy()

        # metrics
        psnr_pha = psnr(image_true=test_gth_pha, image_test=test_retrieved_pha, data_range=1.0)
        ssim_pha = ssim(im1=test_gth_pha, im2=test_retrieved_pha, data_range=1.0, gaussian_weights=True,
                        sigma=1.5, use_sample_covariance=False)
        rmse_pha = rmse(test_retrieved_pha, test_gth_pha)
        pcc_pha = calculate_pcc(test_retrieved_pha, test_gth_pha)

        sum_psnr_pha += psnr_pha
        sum_ssim_pha += ssim_pha
        sum_rmse_pha += rmse_pha
        sum_pcc_pha += pcc_pha

        # show result
        test_gth_pha_rad = test_gth_pha * args.phase_normalize
        test_retrieved_pha_rad = test_retrieved_pha * args.phase_normalize
        show_result(
            save_path=os.path.join(saving_path, f'test{nn + 1}.png'),
            result_data=[
                test_mea_inten,
                test_gth_pha_rad,
                test_retrieved_pha_rad,
                test_gth_aber,
                test_retrieved_aber,
                psnr_pha,
                ssim_pha,
                rmse_pha,
                pcc_pha,
            ],
            args=args,
        )

        if (nn + 1) % N_val == 0:
            sum_psnr_pha = round((sum_psnr_pha) / N_val, 4)
            sum_ssim_pha = round((sum_ssim_pha) / N_val, 4)
            sum_rmse_pha = round((sum_rmse_pha) / N_val, 4)
            sum_pcc_pha = round((sum_pcc_pha) / N_val, 4)

            print(
                f"[PHA] PSNR: {sum_psnr_pha:.4f}, "
                f"SSIM: {sum_ssim_pha:.4f}, "
                f"RMSE: {sum_rmse_pha:.6f}, "
                f"PCC: {sum_pcc_pha:.4f} *****"
            )
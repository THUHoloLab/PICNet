
#==============================================================================
# This script provides the training implementation of the Physics-Informed Cycle-Consistent Network
# (PICNet) for adaptive aberration correction in quantitative phase microscopy.
#
# Author:   Danlin Xu
# Date:     2025/10/30

#Referene

# [1] J.-Y. Zhu, T. Park, P. Isola, et al., “Unpaired image-to-image translation using cycleconsistent adversarial networks,”
# in Proceedings of the IEEE international conference on computer vision, 2223–2232 (2017).
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
#
# [2] C. Lee, G. Song, H. Kim, et al., “Deep learning based on parameterized physical forward
# model for adaptive holographic imaging with unpaired data,” Nat Mach Intell 5(1), 35–45 (2023).
# https://github.com/csleemooo/Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging

#==============================================================================

import time
from itertools import chain
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

# matplotlib.use('Agg')
set_seed(0)

#=================================================
# main function
#=================================================

if __name__ == '__main__':

    # define parameter
    args = parse_args()

    # environment setting
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available")
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define path for loading data and saving result
    dataset_path = "D:\\xdl\\PICNet\\dataset"
    args.result_root = "D:\\xdl\\PICNet\\result"
    args.experiment = "generator1"
    saving_path = os.path.join(args.result_root, args.experiment)

    # =================================================
    # load dataset
    # =================================================

    # load training data of measured intensity
    train_intensity = QPM_Dataset(
        root=dataset_path,
        data_type=["holo"],
        image_set="traindata",
        transform=transforms.Compose([transforms.ToTensor()]),
        ratio=args.train_intensity_ratio,
    )

    train_intensity_loader = DataLoader(train_intensity, batch_size=args.batch_size, shuffle=True)

    # load training data of ground-truth phase
    train_phase = QPM_Dataset(
        root=dataset_path,
        data_type=["pha"],
        image_set="traindata",
        transform=transforms.Compose([transforms.ToTensor()]),
        ratio=args.train_phase_ratio,
    )
    train_phase_loader = DataLoader(train_phase, batch_size=args.batch_size, shuffle=True)

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

    # number of test dataset
    N_val = test_intensity.__len__()

    # =================================================
    # define network / loss function / optimizer
    # =================================================

    # define physical forward model
    Propagator = Physical_Forward_Model(args).to(device=args.device)  # physical model

    # define phase generator for reconstructing sample phase
    Phase_G = Phase_Generator(num_in=1, num_out=1).to(device=args.device)

    # define aberration generator for reconstructing pupil aberrations
    Aberration_G = Aberration_Generator(out_class=12).to(device=args.device)

    # define the discriminator
    Phase_D = Discriminator().to(device=args.device)  # discriminator

    # optimizer
    op_G = torch.optim.Adam(chain(Phase_G.parameters(), Aberration_G.parameters()), lr=args.lr_gen, betas=(0.5, 0.9))
    op_D = torch.optim.Adam(Phase_D.parameters(), lr=args.lr_disc, betas=(0.5, 0.9))

    # scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(op_G, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(op_D, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)

    # loss
    criterion_cycle = nn.L1Loss().to(device=args.device)
    criterion_feature = Loss_feature().to(device=args.device)

    # initialize loss value
    loss_sum_G, loss_sum_intensity, loss_sum_phase, loss_sum_D, loss_sum_aberration = 0, 0, 0, 0, 0
    loss_G_list, loss_intensity_list, loss_phase_list, loss_aberration_list, loss_D_list = [], [], [], [], []
    sum_time = 0
    k_d = 1

    # =================================================
    # train network
    # =================================================

    for it in range(args.iterations):

        # load network model
        Phase_G.train()
        Aberration_G.train()
        Phase_D.train()

        # load training dataset
        gth_phase = next(iter(train_phase_loader))
        measured_intensity, _ = next(iter(train_intensity_loader))
        gth_phase = center_crop(gth_phase, args.crop_size).to(args.device).float() / (args.phase_normalize)
        measured_intensity = center_crop(measured_intensity, args.crop_size).to(args.device).float()

        epoch_start_time = time.time()

        # translation cycle 1: measurement->object->measurement
        retrieved_phase1 = Phase_G(measured_intensity)
        retrieved_aber_coeff1 = Aberration_G(measured_intensity)
        generated_intensity1 = Propagator(retrieved_phase1, retrieved_aber_coeff1).float()

        # translation cycle 2:object->measurement->object
        gth_aber_coeff = generate_zernike(batch_size=args.batch_size, device=args.device)
        generated_intensity2 = Propagator(gth_phase, gth_aber_coeff).float()
        retrieved_phase2 = Phase_G(generated_intensity2)
        retrieved_aber_coeff2 = Aberration_G(generated_intensity2)

        # train discriminator
        ave_critic_loss = 0
        for _ in range(k_d):

            op_D.zero_grad()
            real_D = Phase_D(gth_phase)
            fake_D = Phase_D(retrieved_phase1.detach())
            loss_D = Adversarial_loss_D(real_D, fake_D)

            if torch.isnan(loss_D):
                print("NaN detected in loss")
                break

            ave_critic_loss += loss_D.item() / k_d
            loss_D.backward()
            op_D.step()

        loss_sum_D += ave_critic_loss

        # train generators
        op_G.zero_grad()
        fake_G = Phase_D(retrieved_phase1)
        G_loss = args.adv_coefficient * Adversarial_loss_G(fake_G)
        loss_pha = args.pha_coefficient * criterion_cycle(retrieved_phase2, gth_phase)
        loss_feature_pha = args.feature_pha_coefficient * criterion_feature(retrieved_phase2, gth_phase)
        loss_aber = args.aberration_coefficient * criterion_cycle(retrieved_aber_coeff2, gth_aber_coeff)
        loss_intensity = args.intensity_coefficient * criterion_feature(torch.sqrt(generated_intensity1 + 1e-8),
                                                                        torch.sqrt(measured_intensity + 1e-8))
        loss_G = G_loss + loss_pha + loss_feature_pha + loss_aber + loss_intensity

        if torch.isnan(loss_G):
            print("NaN detected in loss")
            break

        loss_G.backward()
        op_G.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        loss_sum_G += G_loss.item()
        loss_sum_intensity += loss_intensity.item()
        loss_sum_phase += loss_pha.item() + loss_feature_pha.item()
        loss_sum_aberration += loss_aber.item()
        sum_time += epoch_duration

        if (it + 1) % args.chk_iter == 0:

            # print current learning rate
            current_lr_G = lr_scheduler_G.optimizer.param_groups[0]['lr']
            current_lr_D = lr_scheduler_D.optimizer.param_groups[0]['lr']

            # update learning rate
            lr_scheduler_G.step()
            lr_scheduler_D.step()

            # average loss
            loss_sum_G = round(loss_sum_G/args.chk_iter, 4)
            loss_sum_intensity = round(loss_sum_intensity/args.chk_iter, 4)
            loss_sum_phase = round(loss_sum_phase / args.chk_iter, 4)
            loss_sum_aberration = round(loss_sum_aberration / args.chk_iter, 4)
            loss_sum_D = round(loss_sum_D/args.chk_iter, 4)
            round_time = format(sum_time, '.2f')

            # print loss
            print(
                f"[train: {it + 1}/{args.iterations} iter "
                f"| {(it + 1) / args.chk_iter:.2f} epoch "
                f"| {round_time} time] : "
                f"L_G: {loss_sum_G:.6f}, "
                f"L_intensity: {loss_sum_intensity:.6f}, "
                f"L_phase: {loss_sum_phase:.6f}, "
                f"L_aberration: {loss_sum_aberration:.6f}, "
                f"L_D: {loss_sum_D:.6f}, "
                f"lr_G: {current_lr_G:.6f}, "
                f"lr_D: {current_lr_D:.6f}"
            )

            loss_G_list.append(loss_sum_G)
            loss_intensity_list.append(loss_sum_intensity)
            loss_phase_list.append(loss_sum_phase)
            loss_aberration_list.append(loss_sum_aberration)
            loss_D_list.append(loss_sum_D)

            loss_sum_G, loss_sum_intensity, loss_sum_phase, loss_sum_D, loss_sum_aberration = 0, 0, 0, 0, 0

            # make path for saving training parameters
            make_path(saving_path)
            make_path(os.path.join(saving_path, 'generated'))
            p = os.path.join(saving_path, 'generated', 'iterations_' + str(it + 1))
            make_path(p)

            loss = {}
            loss['G_adversarial_loss'] = loss_G_list
            loss['D_adversarial_loss'] = loss_D_list
            loss['G_cycle_intensity_loss'] = loss_intensity_list
            loss['G_cycle_phase_loss'] = loss_phase_list
            loss['G_cycle_aberration_loss'] = loss_aberration_list

            save_data = {'iteration': it + 1,
                         'Phase_G_state_dict': Phase_G.state_dict(),
                         'Phase_D_state_dict': Phase_D.state_dict(),
                         'Aberration_G_state_dict': Aberration_G.state_dict(),
                         'loss': loss,
                         'args': args}

            torch.save(save_data, os.path.join(p, "model.pth"))

            # =================================================
            # test network
            # =================================================

            if (it + 1) % (args.model_chk_iter) == 0:

                # visualize the reconstruction
                with ((torch.no_grad())):

                    # load network model
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
                            save_path=os.path.join(p, f"test{nn + 1}.png"),
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
                            ]
                        )

                        if (nn + 1) % N_val == 0:

                            sum_psnr_pha = round((sum_psnr_pha) / N_val, 4)
                            sum_ssim_pha = round((sum_ssim_pha) / N_val, 4)
                            sum_rmse_pha = round((sum_rmse_pha) / N_val, 4)
                            sum_pcc_pha = round((sum_pcc_pha) / N_val, 4)

                            print(
                                f"***** [val: {it + 1}/{args.iterations} iter | "
                                f"{(it + 1) / args.chk_iter:.2f} epoch] ***** "
                                f"[PHA] PSNR: {sum_psnr_pha:.4f}, "
                                f"SSIM: {sum_ssim_pha:.4f}, "
                                f"RMSE: {sum_rmse_pha:.6f}, "
                                f"PCC: {sum_pcc_pha:.4f} *****"
                            )

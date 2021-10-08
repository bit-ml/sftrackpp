import torch
from torch import nn


MARGIN_MIN = 0.05
MARGIN_MAX = 0.05
EPSILON = 0.0001


def filt2D_batch(conv_filter, matrix):
    return conv_filter(matrix[:, None])[:, 0]


def filt3D_batch(conv_filter, matrix):
    return conv_filter(matrix)


def gauss_3D(kernel_size):
    gauss_kernel = torch.zeros(kernel_size)
    M0 = kernel_size[0] // 2
    M1 = kernel_size[1] // 2
    M2 = kernel_size[2] // 2

    ks0, ks1, ks2 = kernel_size
    std0 = (ks0 - 1) / 4
    std1 = (ks1 - 1) / 4
    std2 = (ks2 - 1) / 4

    mean = torch.Tensor([M0, M1, M2]).float()
    cov_matrix = torch.diag(
        torch.Tensor([1.0 / std0**2, 1.0 / std1**2, 1.0 / std2**2]))

    for i in range(ks0):
        for j in range(ks1):
            for k in range(ks2):
                dx = torch.Tensor([i, j, k]) - mean
                gauss_kernel[i, j, k] = (dx @ cov_matrix) @ dx
    gauss_kernel3D = torch.exp(-0.5 * gauss_kernel)
    return gauss_kernel3D


def init_cnn_filter(kernel_size):
    M0 = kernel_size[0] // 2
    M1 = kernel_size[1] // 2
    M2 = kernel_size[2] // 2

    filter3D = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=(0, M1, M2),
        bias=False,
    )

    kernel3D = gauss_3D(kernel_size)
    kernel3D /= kernel3D.sum()

    filter3D.weight.data.copy_(kernel3D)
    filter3D.weight.requires_grad = False
    return filter3D


def pi_iter_sfseg_batch(sfseg_params, Sp_X, Sp, X, F, F_2, M0):
    alpha, conv_filter = sfseg_params["alpha"], sfseg_params["conv_filter"]

    a0 = filt2D_batch(conv_filter, Sp_X)
    a = a0 - alpha * a0 * F_2[:, M0:-M0]
    b = -filt3D_batch(conv_filter, (F_2 * Sp_X)[:, None])[:, 0]
    c = 2 * F[:, M0:-M0] * filt3D_batch(conv_filter, (F * Sp_X)[:, None])[:, 0]
    result0 = a + alpha * (b + c)
    result = Sp[:, M0:-M0] * result0
    return result


def one_iter_pi_batch(solution, input_orig, init_features, sfseg_params):
    # solution shape: BS x DT X H X W
    p, M0 = sfseg_params["p"], sfseg_params["M0"]

    p_norm = 2
    bs = solution.shape[0]
    num_frames = solution.shape[1]
    guard = 2 * M0

    aux1_input_masks = solution.clone()

    solution_tmp = solution.clone()
    features = init_features

    for frame_idx in range(guard, num_frames - guard):
        s_idx_mw, e_idx_mw = frame_idx - guard, frame_idx + guard

        ############ Power Iteration ############
        F = features[:, s_idx_mw:e_idx_mw + 1]
        X = solution_tmp[:, s_idx_mw:e_idx_mw + 1]
        Sp = input_orig[:, s_idx_mw:e_idx_mw + 1]**p

        Sp_X = Sp * X
        F_2 = F**2

        # one step
        aux1_input_masks[:, frame_idx] = pi_iter_sfseg_batch(
            sfseg_params, Sp_X, Sp, X, F, F_2, M0)[:, M0]
        del Sp_X, F_2

    # divide each frame by frame norm - diferentiable
    frame_norm = (aux1_input_masks.view(bs, num_frames,
                                        -1).norm(p=p_norm, dim=2).view(
                                            bs, num_frames, 1, 1))
    aux2_input_masks = aux1_input_masks / (frame_norm + EPSILON)

    # normalize per frame - diferentiable
    t_min = aux2_input_masks.view(bs, num_frames,
                                  -1).min(dim=2)[0].view(bs, num_frames, 1, 1)
    t_max = ((aux2_input_masks - t_min).view(bs, num_frames,
                                             -1).max(dim=2)[0].view(
                                                 bs, num_frames, 1, 1))

    aux3_input_masks = (aux2_input_masks - t_min) / (t_max + EPSILON)
    solution.copy_(aux3_input_masks)

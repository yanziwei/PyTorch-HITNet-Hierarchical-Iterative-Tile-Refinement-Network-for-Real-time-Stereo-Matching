import sys
sys.path.append('./build')
import array_op
import numpy as np
from utils.read_pfm import pfm_imread
from utils.write_pfm import write_pfm
import pdb
import random
import os
import math
import skimage.io as img_io


def ransac_fit_plane_sf_full_res(src_fn, dst_fp, rs_sigma=1, rs_reg_win=9, rs_nsample=30, rs_iter=10):
    """
    GPU-RANSAC fitting plane for patched in a image on full resolution sceneflow dataset

    :param src_fn: source image name
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_reg_win: regression window (in HITNet, 9x9)
    :param rs_nsample: number of sampled points in a regression window: in general, 3/8 of points in a window
    :param rs_iter: ransac ieration num
    :return: Full resolution dx, dy in pfm format
    """
    # Param init
    data_img, scale = pfm_imread(src_fn)
    data_img = np.ascontiguousarray(data_img, dtype=np.float32)[None, :, :]  # double corresponding to float64
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = data_img.shape[1]
    img_w = data_img.shape[2]
    reg_win = rs_reg_win
    reg_win_rad = (reg_win - 1) // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    total_pixels = (img_w - reg_win_rad * 2) * (img_h - reg_win_rad * 2)

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad + 1):
        for j in range(-reg_win_rad, reg_win_rad + 1):
            if i == -reg_win_rad and j == -reg_win_rad:
                data = data_img[:, :-reg_win_rad + i, :-reg_win_rad + j]
            elif i == reg_win_rad and j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:]), axis=0)
            elif i == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:-reg_win_rad + j]), axis=0)
            elif j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:]), axis=0)
            else:
                data = np.concatenate(
                    (data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:-reg_win_rad + j]), axis=0)
    data = np.array(data)
    data = data.transpose(1, 2, 0)
    # pdb.set_trace()

    # Generate coordinate for a regression window
    x_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    y_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))
    # pdb.set_trace()

    # container array for plane coefficients, e.g. dx, dy and d
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()
    # pdb.set_trace()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_dx = coef[:, 0]
    est_dy = coef[:, 1]
    # est_d = coef[:, 2]

    # reshape and edge padding for estimated results
    est_dx = est_dx.reshape((img_h - reg_win_rad * 2), -1)
    est_dy = est_dy.reshape((img_h - reg_win_rad * 2), -1)
    # est_d = est_d.reshape((img_h - reg_win_rad * 2), -1)
    est_dy_image = np.pad(est_dy, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    est_dx_image = np.pad(est_dx, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    # est_d_image = np.pad(est_d, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    # pdb.set_trace()
    dx_path = os.path.join(dst_fp, 'dx')
    dy_path = os.path.join(dst_fp, 'dy')
    os.makedirs(dx_path, exist_ok=True)
    os.makedirs(dy_path, exist_ok=True)
    write_pfm(dx_path+'/'+src_fn.split('/')[-1], est_dx_image)
    write_pfm(dy_path+'/'+src_fn.split('/')[-1], est_dy_image)
    # write_pfm('est_d_0751.pfm', est_d_image)


def ransac_fit_plane_sf_t2s1(src_fn, dst_fp, rs_sigma=1, rs_reg_win=10, rs_nsample=40, rs_iter=10):
    """
    GPU-RANSAC fitting plane for patched in a image on full resolution with tile size 2x2 on sceneflow dataset

    :param src_fn: source image name
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_reg_win: regression window (in HITNet, 9x9)
    :param rs_nsample: number of sampled points in a regression window: in general, 3/8 of points in a window
    :param rs_iter: ransac ieration num
    :return: 1/2 resolution (ceil(img_size/2)) dx, dy in pfm format
    """
    # Param init
    data_img, scale = pfm_imread(src_fn)
    data_img = np.ascontiguousarray(data_img, dtype=np.float32)[None, :, :]  # double corresponding to float64
    # if image size is not multiple of 2, pad it
    if data_img.shape[1] % 2 != 0:
        data_img = np.pad(data_img, pad_width=((0, 0), (0, 1), (0, 0)), mode='zero')
    if data_img.shape[2] % 2 != 0:
        data_img = np.pad(data_img, pad_width=((0, 0), (0, 0), (0, 1)), mode='zero')
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = data_img.shape[1]
    img_w = data_img.shape[2]
    reg_win = rs_reg_win
    reg_win_rad = reg_win // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    full_coef_h = img_h - 2 * (reg_win_rad - 1)
    full_coef_w = img_w - 2 * (reg_win_rad - 1)

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad):
        for j in range(-reg_win_rad, reg_win_rad):
            # print(i, j, data_img[:, reg_win_rad+i:reg_win_rad+i+full_coef_h, reg_win_rad+j:reg_win_rad+j+full_coef_w].shape)
            if i == -reg_win_rad and j == -reg_win_rad:
                data = data_img[:, :full_coef_h:2, :full_coef_w:2]
            else:
                # pdb.set_trace()
                data = np.concatenate((data, data_img[:, reg_win_rad+i:reg_win_rad+i+full_coef_h:2, reg_win_rad+j:reg_win_rad+j+full_coef_w:2]), axis=0)

    data = np.array(data)
    # pdb.set_trace()
    # data = data[:, ::2, ::2]  # stride the sample points map due to tile size 2
    data = data.transpose(1, 2, 0)
    coef_h = data.shape[0]
    coef_w = data.shape[1]
    total_pixels = coef_h * coef_w
    # pdb.set_trace()

    # Generate coordinate for a regression window. For even size window, coordinated are not exactly the grid index
    x_samples = np.arange(start=-reg_win_rad+0.5, stop=reg_win_rad+0.5, step=1, dtype=np.float32)
    y_samples = np.arange(start=-reg_win_rad+0.5, stop=reg_win_rad+0.5, step=1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))
    # pdb.set_trace()

    # container array for plane coefficients, e.g. dx, dy and d
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()
    # pdb.set_trace()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_dx = coef[:, 0]
    est_dy = coef[:, 1]
    # est_d = coef[:, 2]

    # reshape and edge padding for estimated results
    post_pad = (reg_win_rad - 1) // 2
    est_dx = est_dx.reshape(coef_h, coef_w)
    est_dy = est_dy.reshape(coef_h, coef_w)
    # est_d = est_d.reshape((img_h - reg_win_rad * 2), -1)
    est_dy_image = np.pad(est_dy, pad_width=((post_pad, post_pad), (post_pad, post_pad)), mode='edge')
    est_dx_image = np.pad(est_dx, pad_width=((post_pad, post_pad), (post_pad, post_pad)), mode='edge')
    # est_d_image = np.pad(est_d, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    # pdb.set_trace()
    dx_path = os.path.join(dst_fp, 'dx')
    dy_path = os.path.join(dst_fp, 'dy')
    os.makedirs(dx_path, exist_ok=True)
    os.makedirs(dy_path, exist_ok=True)
    write_pfm(dx_path+'/'+src_fn.split('/')[-1], est_dx_image)
    write_pfm(dy_path+'/'+src_fn.split('/')[-1], est_dy_image)
    # write_pfm('est_d_0751.pfm', est_d_image)


def ransac_fit_plane_sf_tn(src_fn, dst_fp, rs_sigma=1, rs_nsample=40, rs_iter=10, ts=4):
    """
    GPU-RANSAC fitting plane for patched in a image on Nx resolution with tile size NxN(even) on sceneflow dataset
    regression window 10x10
    FIRST PAD TO FIT WINDOWSIZE REGRESSION

    :param src_fn: source image name: source image is already maxpooled
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_nsample: number of sampled points in a regression window
    :param rs_iter: ransac ieration num
    :return: 1/ts resolution (ceil(img_size/2)) dx, dy in pfm format
    """
    # Param init
    data_img, scale = pfm_imread(src_fn)
    data_img = np.ascontiguousarray(data_img, dtype=np.float32)[None, :, :]  # double corresponding to float64
    # if image size is not multiple of tile size, pad it
    if data_img.shape[1] % ts != 0:
        data_img = np.pad(data_img, pad_width=((0, 0), (0, ts-data_img.shape[1]%ts), (0, 0)), mode='edge')
    if data_img.shape[2] % ts != 0:
        data_img = np.pad(data_img, pad_width=((0, 0), (0, 0), (0, ts-data_img.shape[2]%ts)), mode='edge')
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = data_img.shape[1]
    img_w = data_img.shape[2]
    reg_win = 10
    reg_win_rad = reg_win // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    reg_pad = (reg_win - ts) // 2
    # full_coef_h = img_h - 2 * reg_pad
    # full_coef_w = img_w - 2 * reg_pad

    # padding to fit regression window
    reg_pad_data_img = np.pad(data_img, pad_width=((0, 0), (reg_pad, reg_pad), (reg_pad, reg_pad)), mode='edge')

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad):
        for j in range(-reg_win_rad, reg_win_rad):
            # print(i, j, data_img[:, reg_win_rad+i:reg_win_rad+i+full_coef_h, reg_win_rad+j:reg_win_rad+j+full_coef_w].shape)
            if i == -reg_win_rad and j == -reg_win_rad:
                data = reg_pad_data_img[:, :img_h:ts, :img_w:ts]
            else:
                # pdb.set_trace()
                data = np.concatenate((data, reg_pad_data_img[:, reg_win_rad+i:reg_win_rad+i+img_h:ts, reg_win_rad+j:reg_win_rad+j+img_w:ts]), axis=0)

    data = np.array(data)
    # pdb.set_trace()
    # data = data[:, ::2, ::2]  # stride the sample points map due to tile size 2
    data = data.transpose(1, 2, 0)
    coef_h = data.shape[0]
    coef_w = data.shape[1]
    total_pixels = coef_h * coef_w
    pdb.set_trace()

    # Generate coordinate for a regression window. For even size window, coordinated are not exactly the grid index
    x_samples = np.arange(start=-reg_win_rad+0.5, stop=reg_win_rad+0.5, step=1, dtype=np.float32)
    y_samples = np.arange(start=-reg_win_rad+0.5, stop=reg_win_rad+0.5, step=1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))
    # pdb.set_trace()

    # container array for plane coefficients, e.g. dx, dy and d
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()
    # pdb.set_trace()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_dx = coef[:, 0]
    est_dy = coef[:, 1]
    # est_d = coef[:, 2]

    # reshape and edge padding for estimated results
    # post_pad = (reg_win_rad - ts // 2) // 2
    est_dx = est_dx.reshape(coef_h, coef_w)
    est_dy = est_dy.reshape(coef_h, coef_w)
    # est_d = est_d.reshape((img_h - reg_win_rad * 2), -1)
    # est_dy_image = np.pad(est_dy, pad_width=((post_pad, post_pad), (post_pad, post_pad)), mode='edge')
    # est_dx_image = np.pad(est_dx, pad_width=((post_pad, post_pad), (post_pad, post_pad)), mode='edge')
    # est_d_image = np.pad(est_d, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    # pdb.set_trace()
    dx_path = os.path.join(dst_fp, 'dx')
    dy_path = os.path.join(dst_fp, 'dy')
    os.makedirs(dx_path, exist_ok=True)
    os.makedirs(dy_path, exist_ok=True)
    write_pfm(dx_path+'/'+src_fn.split('/')[-1], est_dx)
    write_pfm(dy_path+'/'+src_fn.split('/')[-1], est_dy)
    # write_pfm('est_d_0751.pfm', est_d_image)


def ransac_fit_plane_kt_full_res(src_fn, dst_fp, rs_sigma=1, rs_reg_win=9, rs_nsample=30, rs_iter=10):
    """
    GPU-RANSAC fitting plane for patched in a image on full resolution kt dataset

    :param src_fn: source image name
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_reg_win: regression window (in HITNet, 9x9)
    :param rs_nsample: number of sampled points in a regression window: in general, 3/8 of points in a window
    :param rs_iter: ransac ieration num
    :return: Full resolution dx, dy in pfm format
    """
    # Param init
    data_img = img_io.imread(src_fn)
    data_img = np.ascontiguousarray(data_img, dtype=np.float32)[None, :, :] / 256.  # double corresponding to float64
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = data_img.shape[1]
    img_w = data_img.shape[2]
    reg_win = rs_reg_win
    reg_win_rad = (reg_win - 1) // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    total_pixels = (img_w - reg_win_rad * 2) * (img_h - reg_win_rad * 2)

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad + 1):
        for j in range(-reg_win_rad, reg_win_rad + 1):
            if i == -reg_win_rad and j == -reg_win_rad:
                data = data_img[:, :-reg_win_rad + i, :-reg_win_rad + j]
            elif i == reg_win_rad and j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:]), axis=0)
            elif i == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:-reg_win_rad + j]), axis=0)
            elif j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:]), axis=0)
            else:
                data = np.concatenate(
                    (data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:-reg_win_rad + j]), axis=0)
    data = np.array(data)
    data = data.transpose(1, 2, 0)
    # pdb.set_trace()

    # Generate coordinate for a regression window
    x_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    y_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))
    # pdb.set_trace()

    # container array for plane coefficients, e.g. dx, dy and d
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()
    # pdb.set_trace()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_dx = coef[:, 0]
    est_dy = coef[:, 1]
    # est_d = coef[:, 2]

    # reshape and edge padding for estimated results
    est_dx = est_dx.reshape((img_h - reg_win_rad * 2), -1)
    est_dy = est_dy.reshape((img_h - reg_win_rad * 2), -1)
    # est_d = est_d.reshape((img_h - reg_win_rad * 2), -1)
    est_dy_image = np.pad(est_dy, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    est_dx_image = np.pad(est_dx, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    # est_d_image = np.pad(est_d, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    # pdb.set_trace()
    dx_path = os.path.join(dst_fp, 'dx')
    dy_path = os.path.join(dst_fp, 'dy')
    os.makedirs(dx_path, exist_ok=True)
    os.makedirs(dy_path, exist_ok=True)
    write_pfm(dx_path+'/'+src_fn.split('/')[-1].replace('.png', '.pfm'), est_dx_image)
    write_pfm(dy_path+'/'+src_fn.split('/')[-1].replace('.png', '.pfm'), est_dy_image)
    # write_pfm('est_d_0751.pfm', est_d_image)


def ransac_fit_plane_eth3d_full_res(src_fn, dst_fp, rs_sigma=1, rs_reg_win=9, rs_nsample=30, rs_iter=10):
    """
    GPU-RANSAC fitting plane for patched in a image on full resolution eth3d dataset

    :param src_fn: source image name
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_reg_win: regression window (in HITNet, 9x9)
    :param rs_nsample: number of sampled points in a regression window: in general, 3/8 of points in a window
    :param rs_iter: ransac ieration num
    :return: Full resolution dx, dy in pfm format
    """
    # Param init
    data_img, scale = pfm_imread(src_fn)
    data_img = np.ascontiguousarray(data_img, dtype=np.float32)[None, :, :]  # double corresponding to float64
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = data_img.shape[1]
    img_w = data_img.shape[2]
    reg_win = rs_reg_win
    reg_win_rad = (reg_win - 1) // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    total_pixels = (img_w - reg_win_rad * 2) * (img_h - reg_win_rad * 2)

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad + 1):
        for j in range(-reg_win_rad, reg_win_rad + 1):
            if i == -reg_win_rad and j == -reg_win_rad:
                data = data_img[:, :-reg_win_rad + i, :-reg_win_rad + j]
            elif i == reg_win_rad and j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:]), axis=0)
            elif i == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:-reg_win_rad + j]), axis=0)
            elif j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:]), axis=0)
            else:
                data = np.concatenate(
                    (data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:-reg_win_rad + j]), axis=0)
    data = np.array(data)
    data = data.transpose(1, 2, 0)
    # pdb.set_trace()

    # Generate coordinate for a regression window
    x_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    y_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))
    # pdb.set_trace()

    # container array for plane coefficients, e.g. dx, dy and d
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()
    # pdb.set_trace()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_dx = coef[:, 0]
    est_dy = coef[:, 1]
    # est_d = coef[:, 2]

    # reshape and edge padding for estimated results
    est_dx = est_dx.reshape((img_h - reg_win_rad * 2), -1)
    est_dy = est_dy.reshape((img_h - reg_win_rad * 2), -1)
    # est_d = est_d.reshape((img_h - reg_win_rad * 2), -1)
    est_dy_image = np.pad(est_dy, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    est_dx_image = np.pad(est_dx, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    # est_d_image = np.pad(est_d, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    # pdb.set_trace()
    dx_path = os.path.join(dst_fp, 'dx')
    dy_path = os.path.join(dst_fp, 'dy')
    os.makedirs(dx_path, exist_ok=True)
    os.makedirs(dy_path, exist_ok=True)
    write_pfm(dx_path+'/'+src_fn.split('/')[-1].replace('.png', '.pfm'), est_dx_image)
    write_pfm(dy_path+'/'+src_fn.split('/')[-1].replace('.png', '.pfm'), est_dy_image)
    # write_pfm('est_d_0751.pfm', est_d_image)


def ransac_fit_plane_mb_full_res(src_fn, dst_fp, rs_sigma=1, rs_reg_win=9, rs_nsample=30, rs_iter=10):
    """
    GPU-RANSAC fitting plane for patched in a image on full resolution middlebury dataset

    :param src_fn: source image name
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_reg_win: regression window (in HITNet, 9x9)
    :param rs_nsample: number of sampled points in a regression window: in general, 3/8 of points in a window
    :param rs_iter: ransac ieration num
    :return: Full resolution dx, dy in pfm format
    """
    # Param init
    data_img, scale = pfm_imread(src_fn)
    data_img = np.ascontiguousarray(data_img, dtype=np.float32)[None, :, :]  # double corresponding to float64
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = data_img.shape[1]
    img_w = data_img.shape[2]
    reg_win = rs_reg_win
    reg_win_rad = (reg_win - 1) // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    total_pixels = (img_w - reg_win_rad * 2) * (img_h - reg_win_rad * 2)

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad + 1):
        for j in range(-reg_win_rad, reg_win_rad + 1):
            if i == -reg_win_rad and j == -reg_win_rad:
                data = data_img[:, :-reg_win_rad + i, :-reg_win_rad + j]
            elif i == reg_win_rad and j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:]), axis=0)
            elif i == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:, reg_win_rad + j:-reg_win_rad + j]), axis=0)
            elif j == reg_win_rad:
                data = np.concatenate((data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:]), axis=0)
            else:
                data = np.concatenate(
                    (data, data_img[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:-reg_win_rad + j]), axis=0)
    data = np.array(data)
    data = data.transpose(1, 2, 0)
    # pdb.set_trace()

    # Generate coordinate for a regression window
    x_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    y_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))
    # pdb.set_trace()

    # container array for plane coefficients, e.g. dx, dy and d
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()
    # pdb.set_trace()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_dx = coef[:, 0]
    est_dy = coef[:, 1]
    # est_d = coef[:, 2]

    # reshape and edge padding for estimated results
    est_dx = est_dx.reshape((img_h - reg_win_rad * 2), -1)
    est_dy = est_dy.reshape((img_h - reg_win_rad * 2), -1)
    # est_d = est_d.reshape((img_h - reg_win_rad * 2), -1)
    est_dy_image = np.pad(est_dy, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    est_dx_image = np.pad(est_dx, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    # est_d_image = np.pad(est_d, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    # pdb.set_trace()
    dx_path = os.path.join(dst_fp, 'dx')
    dy_path = os.path.join(dst_fp, 'dy')
    os.makedirs(dx_path, exist_ok=True)
    os.makedirs(dy_path, exist_ok=True)
    write_pfm(dx_path+'/'+src_fn.split('/')[-1].replace('.png', '.pfm'), est_dx_image)
    write_pfm(dy_path+'/'+src_fn.split('/')[-1].replace('.png', '.pfm'), est_dy_image)
    # write_pfm('est_d_0751.pfm', est_d_image)

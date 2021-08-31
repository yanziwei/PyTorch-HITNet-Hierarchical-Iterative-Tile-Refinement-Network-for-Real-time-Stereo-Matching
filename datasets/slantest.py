import sys
import re
import numpy as np
import pdb
import random
import os
import math
import skimage.io as img_io

sys.path.append('/home/yanziwei/data/project-test/Stereo_exp/HITnet/slant/RANSAC_slanted_window_fit/build')
import array_op

def ransac_fit_plane_full_res(src_fn, rs_sigma=1, rs_reg_win=9, rs_nsample=30, rs_iter=10):
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
    # data_img = src_fn
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
    # dx_path = os.path.join(dst_fp, 'dx')
    # dy_path = os.path.join(dst_fp, 'dy')
    # os.makedirs(dx_path, exist_ok=True)
    # os.makedirs(dy_path, exist_ok=True)
    # write_pfm(dx_path+'/'+src_fn.split('/')[-1], est_dx_image)
    # write_pfm(dy_path+'/'+src_fn.split('/')[-1], est_dy_image)
    # write_pfm('est_d_0751.pfm', est_d_image)

    dx_path = src_fn.replace("disparity","slant/dx")
    dy_path = src_fn.replace("disparity","slant/dy")
    dxf,_ = os.path.split(dx_path)
    dyf,_ = os.path.split(dy_path)
    os.makedirs(dxf,exist_ok=True)
    os.makedirs(dyf,exist_ok=True)
    write_pfm(dx_path,est_dx_image)
    write_pfm(dy_path,est_dy_image)
    # return est_dx_image, est_dy_image

def pfm_imread(filename):

    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def write_pfm(file, image, scale = 1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image = np.flipud(image)

    image.tofile(file) 
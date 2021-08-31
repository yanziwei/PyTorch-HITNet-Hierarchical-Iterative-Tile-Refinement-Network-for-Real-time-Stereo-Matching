import numpy as np
from scipy.interpolate import NearestNDInterpolator
import pdb
import skimage.measure as sk_measure
import time
from utils.read_pfm import pfm_imread
from utils.write_pfm import write_pfm


def sparse_disp_interp(s_map):
    """
    Nearest interpolate the missing points in a sparse represented disp map
    :param s_map: sparse map
    :return: dense map
    """
    # pdb.set_trace()
    img_h = s_map.shape[0]
    img_w = s_map.shape[1]
    invalid_mask = (s_map == 0)
    valid_mask = (s_map != np.inf)
    valid_indices = np.argwhere(valid_mask)
    X = np.linspace(0, img_w-1, img_w)
    Y = np.linspace(0, img_h-1, img_h)
    all_X, all_Y = np.meshgrid(X, Y)

    valid_values = s_map[valid_mask]
    # pdb.set_trace()
    interp = NearestNDInterpolator(list(zip(valid_indices[:, 0], valid_indices[:, 1])), valid_values)
    # pdb.set_trace()
    d_map = interp(all_Y, all_X)
    return d_map


def sparse_disp_interp_sn(s_map, stride=None):
    """
    Nearest interpolate the missing points in a max-pooled sparse represented disp map
    :param s_map: sparse map
    :param stride: downsample rate
    :return: dense map
    """
    st_time = time.time()
    if stride is not None:
        if s_map.shape[0] % stride != 0:
            s_map = np.pad(s_map, pad_width=((0, stride-s_map.shape[0]%stride), (0, 0)), mode='edge')
        if s_map.shape[1] % stride != 0:
            s_map = np.pad(s_map, pad_width=((0, 0), (0, stride-s_map.shape[1]%stride)), mode='edge')
        s_map = sk_measure.block_reduce(s_map, (stride, stride), np.max)
    # pdb.set_trace()
    img_h = s_map.shape[0]
    img_w = s_map.shape[1]
    invalid_mask = (s_map == 0)
    valid_mask = (s_map != np.inf)
    valid_indices = np.argwhere(valid_mask)
    X = np.linspace(0, img_w-1, img_w)
    Y = np.linspace(0, img_h-1, img_h)
    all_X, all_Y = np.meshgrid(X, Y)

    valid_values = s_map[valid_mask]
    # pdb.set_trace()
    interp = NearestNDInterpolator(list(zip(valid_indices[:, 0], valid_indices[:, 1])), valid_values)
    # pdb.set_trace()
    d_map = interp(all_Y, all_X)
    print('NearestNDInterpolator exec time: {:.4f}'.format(time.time() - st_time))
    return d_map


if __name__ == '__main__':
    import os
    import time
    import skimage.io as img_io

    # stride = 2
    src_data_path = 'DISP GT PATH'
    dst_data_path = 'OUTPUT_PATH'
    filename = 'filenames/mb_trainF.txt'
    with open(filename) as f:
        disp_gt_fn_lines = [line.rstrip() for line in f.readlines()]

    for i, disp_gt in enumerate(disp_gt_fn_lines):
        # if i == 1:
        #     break
        st_time = time.time()
        src_fn = os.path.join(src_data_path, disp_gt)
        dst_fp = os.path.join(dst_data_path, '/'.join(disp_gt.split('/')[1:-1]))
        s_map, ___ = pfm_imread(src_fn)
        # pdb.set_trace()

        d_map = sparse_disp_interp(s_map)

        d_map_path = os.path.join(dst_fp, 'dense')
        os.makedirs(d_map_path, exist_ok=True)
        write_pfm(d_map_path + '/' + src_fn.split('/')[-1], d_map)
        print('{}th finish: '.format(i) + src_fn + '; Time: {:.2f}s'.format(time.time() - st_time))

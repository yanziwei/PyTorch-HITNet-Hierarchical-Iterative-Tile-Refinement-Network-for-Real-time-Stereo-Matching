import os
import random
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms.functional as photometric
from boxx import *
from numpy.lib.function_base import disp
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
with inpkg():
    from datasets.slantest import ransac_fit_plane_full_res
    from datasets.data_io import Path, get_transform, pfm_imread, read_all_lines


class SceneflowDataset(Dataset):
    def __init__(self, datapath,  training):
        self.datapath = datapath
        self.training = training
        if self.training:
            file_list = "/home/yanziwei/data/project-test/Stereo_exp/HITnet/datasets/list/sceneflow_train.list"
        else:
            file_list = "/home/yanziwei/data/project-test/Stereo_exp/HITnet/datasets/list/sceneflow_test.list"
        f = open(file_list, "r")
        self.file_list = f.readlines()
        self.file_list = self.file_list

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def load_dx_dy(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def load_data(self, data_path, current_file):
        A = current_file
        filename = data_path + "frames_finalpass/" + A[0 : len(A) - 1]
        left = Image.open(filename).convert('RGB')
        filename = (
            data_path
            + "frames_finalpass/"
            + A[0 : len(A) - 14]
            + "right/"
            + A[len(A) - 9 : len(A) - 1]
        )
        right = Image.open(filename).convert('RGB')
        filename = data_path + "disparity/" + A[0 : len(A) - 4] + "pfm"
        dispf = filename
        disp_left, dl_scale = pfm_imread(filename)
        filename = (
            data_path
            + "disparity/"
            + A[0 : len(A) - 14]
            + "right/"
            + A[len(A) - 9 : len(A) - 4]
            + "pfm"
        )
        disp_right, dr_scale = pfm_imread(filename)

        # left = np.asarray(left)
        # right = np.asarray(right)
        slant_dxf = dispf.replace("disparity","slant/dx")
        slant_dx,_ = pfm_imread(slant_dxf)
        slant_dyf = dispf.replace("disparity","slant/dy")
        slant_dy,_ = pfm_imread(slant_dyf)
        return left,right,disp_left,disp_right,slant_dx,slant_dy

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        left_img, right_img, disparity, disparityr,dx_gt,dy_gt = self.load_data(self.datapath, self.file_list[index])

        self.dx_gt_filenames = None
        self.dy_gt_filenames = None
        # if self.dx_gt_filenames and self.dy_gt_filenames:  # has disparity slant param ground truth
        #     dx_gt = self.load_dx_dy(os.path.join(self.datapath, self.dx_gt_filenames[index]))
        #     dy_gt = self.load_dx_dy(os.path.join(self.datapath, self.dy_gt_filenames[index]))
        # else:
            # dx_gt = None
            # dy_gt = None   
            # dx_gt, dy_gt = ransac_fit_plane_full_res(disparity)
            # dx_gt = np.ascontiguousarray(dx_gt,dtype=np.float32)
            # dy_gt = np.ascontiguousarray(dy_gt,dtype=np.float32)
        if self.training:
            w,h = left_img.size
            crop_w, crop_h = 576,384  # similar to crops of HITNet paper, but multiple of 64
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            dx_gt = dx_gt[y1:y1 + crop_h, x1:x1 + crop_w]
            dy_gt = dy_gt[y1:y1 + crop_h, x1:x1 + crop_w]

            # photometric augmentation: brightness and contrast perturb
            sym_random_brt = np.random.uniform(0.8, 1.2)
            sym_random_cts = np.random.uniform(0.8, 1.2)
            asym_random_brt = np.random.uniform(0.95, 1.05, size=2)
            asym_random_cts = np.random.uniform(0.95, 1.05, size=2)
            # brightness
            left_img = photometric.adjust_brightness(left_img, sym_random_brt)
            right_img = photometric.adjust_brightness(right_img, sym_random_brt)
            left_img = photometric.adjust_brightness(left_img, asym_random_brt[0])
            right_img = photometric.adjust_brightness(right_img, asym_random_brt[1])
            # contrast
            left_img = photometric.adjust_contrast(left_img, sym_random_cts)
            right_img = photometric.adjust_contrast(right_img, sym_random_cts)
            left_img = photometric.adjust_contrast(left_img, asym_random_cts[0])
            right_img = photometric.adjust_contrast(right_img, asym_random_cts[1])

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img.copy())
            right_img = processed(right_img.copy())
            

            # random patch exchange of right image
            patch_h = random.randint(50, 180)
            patch_w = random.randint(50, 250)
            patch1_x = random.randint(0, crop_h-patch_h)
            patch1_y = random.randint(0, crop_w-patch_w)
            patch2_x = random.randint(0, crop_h-patch_h)
            patch2_y = random.randint(0, crop_w-patch_w)

            img_patch = right_img[:, patch2_x:patch2_x+patch_h, patch2_y:patch2_y+patch_w]
            right_img[:, patch1_x:patch1_x+patch_h, patch1_y:patch1_y+patch_w] = img_patch
            iter = {"left": left_img,
                    "right": right_img,
                    "disparity": torch.from_numpy(disparity.copy()),
                    "dx_gt": torch.from_numpy(dx_gt.copy()),
                    "dy_gt": torch.from_numpy(dy_gt.copy())}
            return iter
        else:
            w, h = left_img.size
            # left_img = left_img.resize((576,384))
            # right_img = right_img.resize((576,384))
            # normalize
            processed = get_transform()
            left_img = processed(left_img.copy())
            right_img = processed(right_img.copy())

            # pad to size 1280x384
            # top_pad = 384 - h
            # right_pad = 1280 - w
            top_pad = 36
            right_pad = 0
            # assert top_pad > 0 and right_pad > 0
            # # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity.copy(),
                        }
            # pad dx and dy gt
            if dx_gt is not None and dy_gt is not None:
                assert len(dx_gt.shape) == 2
                dx_gt = np.lib.pad(dx_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                assert len(dy_gt.shape) == 2
                dy_gt = np.lib.pad(dy_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None and dx_gt is not None and dy_gt is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "dx_gt": dx_gt,
                        "dy_gt": dy_gt}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}


if __name__ =="__main__":
    datapath = "/home/yanziwei/data/project-test/Stereo_exp/HITnet/datafile/sceneflow/"
    sf = SceneflowDataset(datapath=datapath,training=True)
    TrainImgLoader = DataLoader(sf, 8, shuffle=True, num_workers=8, drop_last=False)
    for iter in tqdm(TrainImgLoader):
        # sleep(2)
        tree-iter
        # pass
        # dx = np.asarray(iter["dx_gt"])
        # cv2.imwrite("slantdx.png",cv2.equalizeHist(dx.squeeze(0).astype(np.uint8)))
        # dy = np.asarray(iter["dx_gt"])
        # cv2.imwrite("slantdy.png",cv2.equalizeHist(dy.squeeze(0).astype(np.uint8)))
        # disp_ = np.asarray(iter["disparity"])
        # cv2.imwrite("disp.png",disp_.squeeze(0))
        # sys.exit(0)
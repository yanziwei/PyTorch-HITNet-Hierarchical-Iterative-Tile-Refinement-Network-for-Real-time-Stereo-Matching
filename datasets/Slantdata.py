from cv2 import imwrite
import numpy as np
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, dataloader
from multiprocessing import Pool
from boxx import inpkg

with inpkg():
    from datasets.slantest import ransac_fit_plane_full_res


class Slantdata:
    def __init__(self, disp_path=None) -> None:
        self.disp_path = disp_path
        self.disp_file = glob.glob(disp_path + "/**/left/*.pfm", recursive=True)
        self.slant_file = glob.glob(
            "/data/datasets/sceneflow/slant/dy/**/left/*.pfm", recursive=True
        )
        print(len(self.disp_file))
        if self.slant_file:
            print(len(self.slant_file))
    def __len__(self):
        return len(self.disp_file)

    def get_slant(self):
        slant_to_disp = [
            file_.replace("slant/dy", "disparity") for file_ in self.slant_file
        ]
        latest_disp = list(set(self.disp_file) - set(slant_to_disp))
        print(len(latest_disp))
        # p = Pool(4)
        # p.map(ransac_fit_plane_full_res,latest_disp)
        for i in latest_disp:
            ransac_fit_plane_full_res(i)


if __name__ == "__main__":
    slant = Slantdata("/data/datasets/sceneflow/disparity")
    slant.get_slant()

import glob
import numpy as np
import re
from tqdm import tqdm
from multiprocessing import Pool
import os

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


slantf = glob.glob("/home/yanziwei/data/datasets/sceneflow/slant/**/*.pfm",recursive=True)
print(len(slantf))


def readslant(file_):
    try:
        slant_data = pfm_imread(file_)
    except Exception as e:
        print(file_)
        os.remove(file_)


if __name__ == "__main__":
    p = Pool()
    p.map(readslant,slantf)
    # for si in tqdm(slantf):
    #     slant_data = pfm_imread(si)
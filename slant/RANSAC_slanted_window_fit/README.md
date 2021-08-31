Run **make.sh** to build python library for parallel computation of slanted plane fitting.

**sf_full_plane_gt.sh** is for sf slant gt generation
**mb_full_plane_gt.sh** is for Middlebury
**kt15_full_plane_gt.sh** is for KITTI2015
**kt12_full_plane_gt.sh** is for KITTI2012
**eth3d_full_plane_gt.sh** is for ETH3d

Before running above plane fit scripts, you need to use sparse2dense_nn.py to generate dense groundtruth for sparse disparity maps.


requirements:
python=3.6.8
cuda version=10


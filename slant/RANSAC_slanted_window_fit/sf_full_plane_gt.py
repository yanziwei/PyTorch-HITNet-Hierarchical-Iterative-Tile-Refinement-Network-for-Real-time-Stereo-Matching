from ransac_fit_plane import ransac_fit_plane_sf_full_res
import os
import time


src_data_path = 'DISP GT PATH'
dst_data_path = 'OUTPUT_PATH'
filename = 'filenames/sceneflow_test_gt.txt'
with open(filename) as f:
    disp_gt_fn_lines = [line.rstrip() for line in f.readlines()]

for i, disp_gt in enumerate(disp_gt_fn_lines):
    # os.makedirs(os.path.join(dst_data_path, '/'.join(disp_gt.split('/')[1:-1])), exist_ok=True)
    src_fn = os.path.join(src_data_path, disp_gt)
    dst_fp = os.path.join(dst_data_path, '/'.join(disp_gt.split('/')[1:-1]))
    st_time = time.time()
    ransac_fit_plane_sf_full_res(src_fn, dst_fp)
    print('{}th finish: '.format(i)+src_fn+'; Time: {:.2f}s'.format(time.time() - st_time))



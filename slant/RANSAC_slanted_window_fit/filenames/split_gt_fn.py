import os


filename = 'sceneflow_test.txt'
with open(filename) as f:
    lines = [line.rstrip() for line in f.readlines()]

gt_lines = []
dst_fn = open('sceneflow_test_gt.txt', 'w')
for i, l in enumerate(lines):
    # if i == 2:
    #     break
    gt = l.split(' ')[-1]
    dst_fn.write(gt)
    dst_fn.write('\n')


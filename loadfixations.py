from os.path import dirname, join as pjoin
import scipy.io as sio

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir, 'COCO_train2014_000000000009.mat')
mat_contents = sio.loadmat('./data/fixations/train/COCO_train2014_000000000009.mat')

x = sorted(mat_contents.keys())
gaze = mat_contents['gaze']
print(gaze)

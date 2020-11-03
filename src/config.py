import os

def make(arr):
    for path in arr:
        if not os.path.exists(path):
            os.mkdir(path)

EPOCHS = 100
ACCUMULATE = 128

LR = 0.0001

MODEL_NAME = 'POINTNET_G'

INPUT = os.path.join(os.getcwd(), 'input')
OUTPUT = os.path.join(os.getcwd(), 'output')


TRAIN_BATCH_SIZE = 100
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE


logs_path = os.path.join(os.getcwd(), 'lightning_logs')
version = 'version_0'
ckpt_name = 'epoch=1'

PATH = os.path.join(logs_path, version, 'checkpoints', ckpt_name+'.ckpt')

save_path = os.path.join(OUTPUT, 'saved')

MODEL_LATEST = os.path.join(OUTPUT, 'latest_model.pth')


GPUS = '0'

arr = [INPUT, OUTPUT, logs_path, save_path]
make(arr)


HBACKWARD = 15
HFORWARD = 50
NFRAMES = 10
FRAME_STRIDE = 15
AGENT_FEATURE_DIM = 8
MAX_AGENTS = 150

# TRAIN_ZARR = os.path.join(INPUT, 'data', 'lyft', 'scenes', 'train.zarr')

TRAIN_ZARR = os.path.join(INPUT, 'data', 'lyft', 'lyft_full', 'train_full.zarr')

VALID_ZARR = os.path.join(INPUT, 'data', 'lyft', 'scenes', 'validate.zarr')

TEST_ZARR = os.path.join(INPUT, 'data', 'lyft', 'scenes', 'test.zarr')
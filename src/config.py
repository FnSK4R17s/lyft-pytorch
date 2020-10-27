import os

def make(arr):
    for path in arr:
        if not os.path.exists(path):
            os.mkdir(path)


BUCKETS = [8, 16, 32, 64, 128, 256, 512]
CHANNELS = 1
HEIGHT = 32

CLASSES = 1

INPUT_LENGTH = BUCKETS[-2]

EPOCHS = 7
ACCUMULATE = 32

LR = 0.001

# INPUT_SHAPE = (HEIGHT, CROP_SIZE, CHANNELS)

MODEL_NAME = 'CRNN'


INPUT = os.path.join(os.getcwd(), 'input')
OUTPUT = os.path.join(os.getcwd(), 'output')

TRAIN_PATH_X = os.path.join(INPUT, 'data', 'crops_centernet', 'crops_centernet')
TRAIN_PATH_Y = os.path.join(INPUT, 'data', 'crops_predictions.txt')

LICENSE_SYNTH_X = os.path.join(INPUT, 'license')
LICENSE_SYNTH_Y = os.path.join(INPUT, 'license_synth2.txt')

MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)

TRAIN_BATCH_SIZE = 320
VAL_BATCH_SIZE = 320//2


logs_path = os.path.join(os.getcwd(), 'lightning_logs')
version = 'version_0'
ckpt_name = 'epoch=1'

PATH = os.path.join(logs_path, version, 'checkpoints', ckpt_name+'.ckpt')

ONNX_DYNAMIC = os.path.join(OUTPUT, 'crnn.onnx')

save_path = os.path.join(OUTPUT, 'saved')

MODEL_LATEST = os.path.join(OUTPUT, 'latest_model.pth')

DF_PATH = os.path.join(OUTPUT, 'df.csv')
BUCKETIZED_DF_PATH = os.path.join(OUTPUT, 'df_bucketized.csv')
FOLDED_DF_PATH = os.path.join(OUTPUT, 'df_folded.csv')

folds_3 = list(range(300,302))
folds_4 = list(range(400,430))
folds_5 = list(range(500,542))
folds_6 = list(range(600,603))

# TRAIN_FOLDS = [folds_3, folds_4, folds_5, folds_6]
# TRAIN_FOLDS = list(range(500,542))

TRAIN_FOLDS = folds_4
TRAIN_FOLDS.extend(folds_5)

VAL_FOLDS = list(range(600,603))

TEST_FOLDS = list(range(300,302))

PLT_PATH = os.path.join(OUTPUT, 'img.png')

GPUS = '1, 2'

arr = [INPUT, OUTPUT, logs_path, save_path]
make(arr)


HBACKWARD = 15
HFORWARD = 50
NFRAMES = 10
FRAME_STRIDE = 15
AGENT_FEATURE_DIM = 8
MAX_AGENTS = 150
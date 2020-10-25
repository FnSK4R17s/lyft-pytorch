import albumentations as A
from albumentations.core.composition import OneOf
import numpy as np
import torch, os, math
from tqdm import tqdm
from glob import glob
import config

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

from sklearn.model_selection import KFold
import cv2

from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DATADataset:
    def __init__(self, df, dl_type='val'):
        self.buckets = config.BUCKETS
        self.load_data()
        self.df = df
        self.dl_type = dl_type
    
    def load_data(self):
        pass

    def encode_label(self, label):
        label = list(str(label))
        length = len(label)
        label = config.VOCAB.encoder(label)
        return label, length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.loc[item]
        factor = np.divide(config.HEIGHT, row.h)
        width = int(np.ceil(np.multiply(row.w, factor)))

        bucket_number = -2

        if self.dl_type == 'train':   
            aug = A.Compose([
                A.Resize(config.HEIGHT, width, always_apply=True),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=5,
                    p=0.9),
                A.Blur(p=0.5)
            ])
        elif self.dl_type == 'val':
            bucket_number = -1
            aug = A.Compose([
                A.Resize(config.HEIGHT, width, always_apply=True)
            ])
        else:
            bucket_number = 3
            aug = A.Compose([
                A.Resize(config.HEIGHT, width, always_apply=True)
            ])

        image = np.array(Image.open(row.path).convert('L'))[... , np.newaxis]

        encoded_label, encoded_length = self.encode_label(row.label)
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)
        label = torch.full(size=(1, config.INPUT_LENGTH), fill_value=-1, dtype=torch.long)
        label[:, :encoded_length] = encoded_label
        label = label.squeeze()
        augmented = aug(image=image)

        image = augmented['image']

        
        # padded_image = np.zeros(( config.HEIGHT, self.buckets[bucket_number], 1))
        # padded_image[:config.HEIGHT,:width,:] = image


        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # image = np.transpose(padded_image, (2, 0, 1)).astype(np.float32)

        length = torch.zeros(1, config.INPUT_LENGTH)
        mask = torch.ones(1, encoded_length)
        length[:, :encoded_length] = mask

        return{
            'image' : torch.tensor(image, dtype=torch.float),
            'label' : label,
            'bucket' : torch.tensor([(self.buckets[bucket_number]//4)-1], dtype=torch.long),
            'key' : row.path,
            'length' : torch.tensor(encoded_length, dtype=torch.long)
        }
        

class DATADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size=10, val_batch_size=config.VAL_BATCH_SIZE, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        # print(self.batch_size)

    def prepare_data(self):
        if not os.path.exists(config.DF_PATH):
            self.prepare_df()
        if not os.path.exists(config.BUCKETIZED_DF_PATH):
            self.bucketize()
        if not os.path.exists(config.FOLDED_DF_PATH):
            self.fold_df()
        print("Data preparation done !")

    def prepare_df(self):
        try:
            # df.to_csv(config.DF_PATH, index=False)
            print('Saved csv file to ', config.DF_PATH)
        except:
            print('An error occured while saving the Dataframe')
        
    def bucketize(self):
        print('df.csv file found, bucketizing...')

    def fold_df(self):
        print('df_bucketized.csv file found, creating folds...')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        print('Reading Folds CSV ...')
        df = pd.read_csv(config.FOLDED_DF_PATH)
        if stage == 'fit' or stage is None:
            # print(config.TRAIN_FOLDS)
            self.df_train = df[df.kfold.isin(config.TRAIN_FOLDS)].reset_index(drop=True)
            # self.df_train = df[df.kfold.isin([500])].reset_index(drop=True)

            self.df_val = df[df.kfold.isin(config.VAL_FOLDS)].reset_index(drop=True)
            # self.df_val = df[df.kfold.isin([600])].reset_index(drop=True)

        if stage == 'test' or stage is None:
            # print(config.TEST_FOLDS)
            self.df_train = df[df.kfold.isin(config.TEST_FOLDS)].reset_index(drop=True)


    def train_dataloader(self):
        return DataLoader(DATADataset(self.df_train, dl_type='train'), shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(DATADataset(self.df_val, dl_type='val'), batch_size=self.val_batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(DATADataset(self.df_train, dl_type='test'), batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def collate_fn(self, batch):
        image = []
        label = []
        bucket = []
        key = []
        length = []
        for item in batch:
            image.append(item['image'])
            label.append(item['label'])
            bucket.append(item['bucket'])
            key.append(item['key'])
            length.append(item['length'])
        
        return{
            'image' : torch.stack(image),
            'label' : torch.stack(label),
            'bucket' : torch.stack(bucket),
            'key' : key,
            'length' : torch.stack(length)
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random

    x = 0
    y = 0

    dm = DATADataModule()

    dm.prepare_data()

    dm.setup('fit')
    for batch in dm.train_dataloader():
        # A, *_ = batch
        # print(batch)
        x = batch['image'][0].numpy()
        print(x)
        print(x.shape)
        y = batch['label'][0].squeeze().numpy()
        print(y)
        print(y.shape)
        print(config.VOCAB.decoder(y))
        l = batch['length'][0]
        print(l)
        # plt.imshow(x)
        # plt.show()
        break
    # dm.setup('test')

    # data = OCRDataset()
    # print(len(data))
    # idx = random.choice(range(len(data)))
    # datapoint = data[idx]
    
    # img = datapoint['image'].squeeze().numpy()
    # label = datapoint['label'].numpy()
    # bucket = datapoint['bucket']
    # key = datapoint['key']
    # length = datapoint['length']


    # print(idx, key, bucket)
    # print(length)

    # print(img.shape)
    # print(config.VOCAB.decoder(label))

    # plt.imshow(img)

    # plt.show()

    img = x
    mask = y

    I = np.transpose(img, (1, 2, 0))

    I8 = (((I - I.min()) / (I.max() - I.min())) * 255.0).astype(np.uint8)

    img = Image.fromarray(I8[:,:,0])
    img.save(config.PLT_PATH)
    # norm = plt.Normalize(vmin=img.min(), vmax=img.max())
    # img = norm(img)
    # plt.imsave(config.PLT_PATH, img)

    # plt.imsave(config.MASK_PATH, mask)
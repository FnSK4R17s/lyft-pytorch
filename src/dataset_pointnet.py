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


PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_NOT_SET",
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_DONTCARE",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
    "AVRESEARCH_LABEL_DONTCARE",
]
KEPT_PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
]
KEPT_PERCEPTION_LABELS_DICT = {label:PERCEPTION_LABELS.index(label) for label in KEPT_PERCEPTION_LABELS}
KEPT_PERCEPTION_KEYS = sorted(KEPT_PERCEPTION_LABELS_DICT.values())


class LabelEncoder:
    def  __init__(self, max_size=500, default_val=-1):
        self.max_size = max_size
        self.labels = {}
        self.default_val = default_val

    @property
    def nlabels(self):
        return len(self.labels)

    def reset(self):
        self.labels = {}

    def partial_fit(self, keys):
        nlabels = self.nlabels
        available = self.max_size - nlabels

        if available < 1:
            return

        keys = set(keys)
        new_keys = list(keys - set(self.labels))

        if not len(new_keys):
            return
        
        self.labels.update(dict(zip(new_keys, range(nlabels, nlabels + available) )))
    
    def fit(self, keys):
        self.reset()
        self.partial_fit(keys)

    def get(self, key):
        return self.labels.get(key, self.default_val)
    
    def transform(self, keys):
        return np.array(list(map(self.get, keys)))

    def fit_transform(self, keys, partial=True):
        self.partial_fit(keys) if partial else self.fit(keys)
        return self.transform(keys)


class CustomLyftDataset(Dataset):
    feature_mins = np.array([-17.336, -27.137, 0. , 0., 0. , -3.142, -37.833, -65.583],
    dtype="float32")[None,None, None]

    feature_maxs = np.array([17.114, 20.787, 42.854, 42.138,  7.079,  3.142, 29.802, 35.722],
    dtype="float32")[None,None, None]



    def __init__(self, zdataset, scenes=None, nframes=10, frame_stride=15, hbackward=10, 
                 hforward=50, max_agents=150, agent_feature_dim=8):
        """
        Custom Lyft dataset reader.
        
        Parmeters:
        ----------
        zdataset: zarr dataset
            The root dataset, containing scenes, frames and agents
            
        nframes: int
            Number of frames per scene
            
        frame_stride: int
            The stride when reading the **nframes** frames from a scene
            
        hbackward: int
            Number of backward frames from  current frame
            
        hforward: int
            Number forward frames from current frame
        
        max_agents: int 
            Max number of agents to read for each target frame. Note that,
            this also include the backward agents but not the forward ones.
        """
        super().__init__()
        self.zdataset = zdataset
        self.scenes = scenes if scenes is not None else []
        self.nframes = nframes
        self.frame_stride = frame_stride
        self.hbackward = hbackward
        self.hforward = hforward
        self.max_agents = max_agents

        self.nread_frames = (nframes-1)*frame_stride + hbackward + hforward

        self.frame_fields = ['timestamp', 'agent_index_interval']

        self.agent_feature_dim = agent_feature_dim

        self.filter_scenes()
      
    def __len__(self):
        return len(self.scenes)

    def filter_scenes(self):
        self.scenes = [scene for scene in self.scenes if self.get_nframes(scene) > self.nread_frames]


    def __getitem__(self, index):
        return self.read_frames(scene=self.scenes[index])

    def get_nframes(self, scene, start=None):
        frame_start = scene["frame_index_interval"][0]
        frame_end = scene["frame_index_interval"][1]
        nframes = (frame_end - frame_start) if start is None else ( frame_end - max(frame_start, start) )
        return nframes


    def _read_frames(self, scene, start=None):
        nframes = self.get_nframes(scene, start=start)
        assert nframes >= self.nread_frames

        frame_start = scene["frame_index_interval"][0]

        start = start or frame_start + np.random.choice(nframes-self.nread_frames)
        frames = self.zdataset.frames.get_basic_selection(
            selection=slice(start, start+self.nread_frames),
            fields=self.frame_fields,
            )
        return frames
    

    def parse_frame(self, frame):
        return frame

    def parse_agent(self, agent):
        return agent

    def read_frames(self, scene, start=None,  white_tracks=None, encoder=False):
        white_tracks = white_tracks or []
        frames = self._read_frames(scene=scene, start=start)

        agent_start = frames[0]["agent_index_interval"][0]
        agent_end = frames[-1]["agent_index_interval"][1]

        agents = self.zdataset.agents[agent_start:agent_end]


        X = np.zeros((self.nframes, self.max_agents, self.hbackward, self.agent_feature_dim), dtype=np.float32)
        target = np.zeros((self.nframes, self.max_agents, self.hforward, 2),  dtype=np.float32)
        target_availability = np.zeros((self.nframes, self.max_agents, self.hforward), dtype=np.uint8)
        X_availability = np.zeros((self.nframes, self.max_agents, self.hbackward), dtype=np.uint8)

        for f in range(self.nframes):
            backward_frame_start = f*self.frame_stride
            forward_frame_start = f*self.frame_stride+self.hbackward
            backward_frames = frames[backward_frame_start:backward_frame_start+self.hbackward]
            forward_frames = frames[forward_frame_start:forward_frame_start+self.hforward]

            backward_agent_start = backward_frames[-1]["agent_index_interval"][0] - agent_start
            backward_agent_end = backward_frames[-1]["agent_index_interval"][1] - agent_start

            backward_agents = agents[backward_agent_start:backward_agent_end]

            le = LabelEncoder(max_size=self.max_agents)
            le.fit(white_tracks)
            le.partial_fit(backward_agents["track_id"])

            for iframe, frame in enumerate(backward_frames):
                backward_agent_start = frame["agent_index_interval"][0] - agent_start
                backward_agent_end = frame["agent_index_interval"][1] - agent_start

                backward_agents = agents[backward_agent_start:backward_agent_end]

                track_ids = le.transform(backward_agents["track_id"])
                mask = (track_ids != le.default_val)
                mask_agents = backward_agents[mask]
                mask_ids = track_ids[mask]
                X[f, mask_ids, iframe, :2] = mask_agents["centroid"]
                X[f, mask_ids, iframe, 2:5] = mask_agents["extent"]
                X[f, mask_ids, iframe, 5] = mask_agents["yaw"]
                X[f, mask_ids, iframe, 6:8] = mask_agents["velocity"]

                X_availability[f, mask_ids, iframe] = 1

            
            for iframe, frame in enumerate(forward_frames):
                forward_agent_start = frame["agent_index_interval"][0] - agent_start
                forward_agent_end = frame["agent_index_interval"][1] - agent_start

                forward_agents = agents[forward_agent_start:forward_agent_end]

                track_ids = le.transform(forward_agents["track_id"])
                mask = track_ids != le.default_val

                target[f, track_ids[mask], iframe] = forward_agents[mask]["centroid"]
                target_availability[f, track_ids[mask], iframe] = 1

        target -= X[:,:,[-1], :2]
        target *= target_availability[:,:,:,None]
        X[:,:,:, :2] -= X[:,:,[-1], :2]
        X *= X_availability[:,:,:,None]
        X -= self.feature_mins
        X /= (self.feature_maxs - self.feature_mins)

        if encoder:
            return X, target, target_availability, le
        return X, target, target_availability



def collate(x):
    x = map(np.concatenate, zip(*x))
    x = map(torch.from_numpy, x)
    return x

def shapefy( xy_pred, xy, xy_av):
    NDIM = 3
    xy_pred = xy_pred.view(-1, HFORWARD, NDIM, 2)
    xy = xy.view(-1, HFORWARD, 2)[:,:,None]
    xy_av = xy_av.view(-1, HFORWARD)[:,:,None]
    return xy_pred, xy,xy_av

def LyftLoss(c, xy_pred, xy, xy_av):
    c = c.view(-1,c.shape[-1])
    xy_pred, xy, xy_av  = shapefy(xy_pred, xy, xy_av)
    
    c = torch.softmax(c, dim=1)
    
    l = torch.sum(torch.mean(torch.square(xy_pred-xy), dim=3)*xy_av, dim=1)
    
    # The LogSumExp trick for better numerical stability
    # https://en.wikipedia.org/wiki/LogSumExp
    m = l.min(dim=1).values
    l = torch.exp(m[:, None]-l)
    
    l = m - torch.log(torch.sum(l*c, dim=1))
    denom = xy_av.max(2).values.max(1).values
    l = torch.sum(l*denom)/denom.sum()
    return 3*l # I found that my loss is usually 3 times smaller than the LB score


def MSE(xy_pred, xy, xy_av):
    xy_pred, xy, xy_av = shapefy(xy_pred, xy, xy_av)
    return 9*torch.mean(torch.sum(torch.mean(torch.square(xy_pred-xy), 3)*xy_av, dim=1))

def MAE(xy_pred, xy, xy_av):
    xy_pred, xy, xy_av = shapefy(xy_pred, xy, xy_av)
    return 9*torch.mean(torch.sum(torch.mean(torch.abs(xy_pred-xy), 3)*xy_av, dim=1))

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
import torch
import model_dispatcher
import config
import os
import numpy as np
from PIL import Image
import math


model = model_dispatcher.MODELS[config.MODEL_NAME]

model.load_state_dict(torch.load(config.MODEL_SAVE))
model.eval()

image = os.path.join(config.TRAIN_PATH_X, '924_0.jpg')

image = Image.open(image).convert('L')

print(image.size)

w, h = image.size

factor = (32.0/h)
width = int(math.ceil(w*factor))

image = np.array(image.resize((width*4, 32)))[... , np.newaxis]

print(image.shape)

image = np.transpose(image, (2, 0, 1)).astype(np.float32)

image = torch.tensor(image[np.newaxis, ...], dtype=torch.float)

y_hat = model(image).squeeze(0).detach().numpy()

print(y_hat.shape)

print(np.argmax(y_hat, -1).squeeze())

print(config.VOCAB.decoder(np.argmax(y_hat, -1).squeeze()))
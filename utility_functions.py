
import numpy as np
from skimage import io

def show(img):
    if len(img.shape)==4:
        io.imshow(img[0])
    else:
        io.imshow(img)
    io.show()

def read_image(path):
    img = io.imread(path)[:,:,:3]
    smaller_side = np.min(img.shape[:-1])
    img = img[:smaller_side, :smaller_side, :]
    return np.expand_dims(img, axis=0)

def convert_to_real_image(cimg):
    img = cimg[:,:,:,:]
    img = np.squeeze(img, 0)
    img[:, :, 0] = img[:, :, 0] + 103.939
    img[:, :, 1] = img[:, :, 1] + 116.779
    img[:, :, 2] = img[:, :, 2] + 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

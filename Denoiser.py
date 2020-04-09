import numpy as np
from scipy import signal
from PIL import Image


def load_image(path):
    return np.asarray(Image.open(path))

def save(path, img):
    tmp = np.asarray(img, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_image(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, 11)
    save('background.png', bg)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 0.1
    save('foreground_mask.png', mask)

    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, inp, 1.0)


inp_path = 'input path'
out_path = 'output path'

inp = load_image(inp_path)
out = denoise_image(inp)

save(out_path, out)

import os
import numpy as np
from PIL import Image


def load_image(filename):
	img = Image.open(filename)
	img = np.array(img)
	return img
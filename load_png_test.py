from PIL import Image
import numpy as np


pic = Image.open("97/4333.png")
pix = np.array(pic)
print(pix)
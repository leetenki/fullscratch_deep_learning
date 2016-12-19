import sys, os
import numpy as np
from PIL import Image
sys.path.append(os.pardir)

def im_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

img = x_train[0].reshape(28, 28)
im_show(img)

import numpy as np
import os
import sys
from skimage import io

dir_name = sys.argv[1]
your_list = os.listdir(dir_name)

img = []
for jpg in your_list:
    temp_img = io.imread(os.path.join(dir_name, jpg))
    temp_img = temp_img.flatten()
    img.append(temp_img)

img = np.array(img)

target = sys.argv[2]
idx = int(target[:target.find('.')])

print(img.shape)
print(idx)

mu = np.mean(img, axis=0)

img = img - mu
U, S, V = np.linalg.svd(img.T, full_matrices=False)

weights = np.dot(img, U)

k = 4
result = mu + np.dot(weights[idx, 0:k], U[:, 0:k].T)

result -= np.min(result)
result /= np.max(result)
result = (result*255).astype(np.uint8)
io.imsave('reconstruction.jpg', result.reshape((600, 600, 3)))

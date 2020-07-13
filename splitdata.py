import os
import glob
from random import shuffle

images = glob.glob('../images/*')
shuffle(images)

num_images = len(images)
split = int(0.8*num_images)

train_images = images[:split]
test_images = images[:split]
print(train_images,test_images)

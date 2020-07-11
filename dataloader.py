
import os
import glob
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import load_image
import numpy as np


class mntdata(Dataset):
	def __init__(self,opt):
		self.samples = []

		data = open(opt.target_path)
		target_csv = list(csv.reader(data,delimiter=','))

		self.classes = []
		for row in target_csv:
			self.classes.append(row[2])
		self.classes = list(set(self.classes))
		opt.num_classes = len(self.classes)

		self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])	

		
		for row in target_csv[1:]:
			sample_dict= {}
			input_path = os.path.join(opt.data_root,row[0] + '.jpg')
			target = int(self.classes.index(row[2]))

			
			if os.path.isfile(input_path):
				image = load_image(input_path)
				if checkimage(image):
					image = self.transforms(image)
					self.samples.append([[image,target]])

		print('Data Loaded')
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]


def checkimage(img):
	img = np.array(img)
	if len(img.shape) != 3:
		return False
	if img.shape[2] != 3:
		return False
	return True


class mntdataloader(object):
	def __init__(self, opt, dataset):
		super(mntdataloader, self).__init__()

		if opt.shuffle:
			train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
		else:
			train_sampler = None

		self.data_loader = torch.utils.data.DataLoader(
			dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
			num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
		self.dataset = dataset
		self.data_iter = self.data_loader.__iter__()

	def __len__(self):
		return len(self.data_loader)

	def next_batch(self):
		try:
			batch = self.data_iter.__next__()
		except StopIteration:
			self.data_iter = self.data_loader.__iter__()
			batch = self.data_iter.__next__()

		return batch
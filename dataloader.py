
import os
import glob
import torch
from torch.utils.data import Dataset
import csv


class mntdata(Dataset):
	def __init__(self, data_root, target_csv):
		self.samples = []

		data = open(target_csv)
		target_csv = list(csv.reader(data,delimiter=','))

		self.classes = []
		for row in target_csv:
			self.classes.append(row[2])
		self.classes = list(set(self.classes))
		
		
		for row in target_csv:
			sample_dict= {}
			sample_dict['input'] = os.path.join(data_root,row[0] + '.jpg')
			sample_dict['target'] = int(self.classes.index(row[2]))
			if os.path.isfile(sample_dict['input']):
				self.samples.append(sample_dict)

			

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]


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

	def next_batch(self):
		try:
			batch = self.data_iter.__next__()
		except StopIteration:
			self.data_iter = self.data_loader.__iter__()
			batch = self.data_iter.__next__()

		return batch
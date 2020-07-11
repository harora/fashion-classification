import os
import glob
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from config import opt
from dataloader import mntdataloader,mntdata
from models import get_model



def train(**kwargs):
	opt.parse(kwargs)

	trainset = mntdata(opt)
	trainloader = mntdataloader(opt,trainset)
	model = get_model(opt.model_name, in_channels=opt.in_channels, img_rows=opt.img_rows, num_classes=opt.num_classes)
	if opt.gpu:
		model.cuda(0)
		model.load(opt.model_save_path)
	model.train()
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

	for epoch in range(1, opt.max_epochs+1):
		for batch_idx, (data) in enumerate(trainloader.next_batch()):
			
			data, target = data[0],data[1]
			print(data.shape,target)
			if opt.gpu:
				data, target = Variable(data.cuda(0)), Variable(target.cuda(0))
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0]))
		if epoch % opt.save_freq == 0:
			model.save()


if __name__ == '__main__':
	train()
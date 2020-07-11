import os
import glob
from PIL import Image

from config import opt
from dataloader import mntdataloader,mntdata



def train(**kwargs):
	opt.parse(kwargs)

	trainset = mntdata(opt.data_path,opt.target_path)
	train_loader = mntdataloader(opt,trainset)
	model = get_model(opt.model_name, in_channels=opt.in_channels, img_rows=opt.img_rows, num_classes=opt.num_classes)
	# model.cuda(0)
	#model.load(opt.model_save_path)
	model.train()
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

	for epoch in range(1, opt.max_epochs+1):
		for batch_idx, (data, target) in enumerate(train_loader):
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
import os
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
# 准备测试集
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练数据集的模板长度{train_data_size}")
print(f"训练数据集的模板长度{test_data_size}")

dataloder_train = DataLoader(train_data, batch_size=64)
dataloder_test = DataLoader(test_data, batch_size=64)

# 创建网络模型
model = Model()
if torch.cuda.is_available():
	model = model.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 使用交叉商
if torch.cuda.is_available():
	loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数

total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录训练
epoch = 10  # 记录训练的轮数

# 添加tensorboard
writer = SummaryWriter('./logs')

for i in range(epoch):
	print(f"---------第{i + 1}训练开始--------")
	# 训练开始
	model.train()
	for data in dataloder_train:
		img, target = data
		if torch.cuda.is_available():
			img = img.cuda()
			target=target.cuda()
		output = model(img)
		loss = loss_fn(output, target)  # 计算损失
		# 优化调参
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		total_train_step += 1
		if total_train_step % 100 == 0:
			print(f"训练次数{total_train_step},Loss是{loss.item()}")
			writer.add_scalar('train_loss', loss.item(), total_train_step)
	
	# 观察测试数据来评价模型的训练
	model.eval()
	total_test_loss = 0
	total_accuracy = 0
	with torch.no_grad():
		for data in dataloder_test:
			img, target = data
			if torch.cuda.is_available():
				img = img.cuda()
				target=target.cuda()
			output = model(img)
			loss = loss_fn(output, target)
			total_test_loss += loss
			accuracy = (output.argmax(1) == target).sum()
			total_accuracy += accuracy
	
	print(f"整体测试集上的loss是{total_test_loss}")
	print(f"整体测试集上的准确率是{total_accuracy / test_data_size}")
	writer.add_scalar("test_loss", total_test_loss, total_test_step)
	total_test_step += 1
	
	torch.save(model, f"models/model{i + 1}.pth")
	print("模型已保存")

writer.close()


import torch
import torch.nn as nn
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.MaxPool2d(kernel_size=2),
			nn.Flatten(),
			nn.Linear(in_features=1024, out_features=64),
			nn.Linear(in_features=64, out_features=10)
		)
	
	def forward(self, x):
		return self.model(x)

if __name__=='__main__':
	model=Model()
	input=torch.ones((64,3,32,32))
	output=model(input)
	print(output.shape)
	
from  PIL import Image
import torch
import torch.nn as nn
import torchvision

img_path='./imgs/005.png'
image=Image.open(img_path)
image=image.convert('RGB')

transforms=torchvision.transforms.Compose(
	[
		torchvision.transforms.Resize((32,32)),
		torchvision.transforms.ToTensor()
	]
)

image=transforms(image)
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
#加载网络模型
model=torch.load("./models/model30.pth",map_location=torch.device('cpu'))#将GPU上处理的图片放在CPU上使用

image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
	output=model(image)
	print(output)

object_dict={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
print(object_dict[int(output.argmax(1))])

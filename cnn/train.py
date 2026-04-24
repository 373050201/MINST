from torch.utils.data import random_split, DataLoader
from model import MyModel
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch



#加载训练集并分割为训练集和验证集
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])
data=torchvision.datasets.MNIST(root="../dataset",train=True,transform=transform_train)
train_size=int(0.9*len(data))
val_size=len(data)-train_size
train_data,val_data=random_split(data,[train_size,val_size],torch.Generator().manual_seed(42))
train_loader=DataLoader(train_data,batch_size=64,drop_last=True)
val_loader=DataLoader(val_data,batch_size=64,drop_last=True)

print("正在初始化...")
myModel=MyModel()#模型
myModel=myModel.cuda()
criterion = nn.CrossEntropyLoss()#损失函数
criterion=criterion.cuda()
optimizer=torch.optim.Adam(myModel.parameters(),lr=0.001)#优化器

total_train=0#总训练次数
epoch=10#训练轮次
for i in range(epoch):
    print(f"第{i+1}轮训练开始...")
    for item in train_loader:
        imgs,targets=item
        imgs=imgs.cuda()
        targets=targets.cuda()

        outputs=myModel(imgs)
        loss=criterion(outputs,targets)

        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#优化器优化
        total_train+=1
    print(f"此轮已训练了{total_train}次")

    total_loss=0
    total_correct=0
    with torch.no_grad():
        for item in val_loader:
            imgs,targets=item
            imgs=imgs.cuda()
            targets=targets.cuda()

            outputs=myModel(imgs)
            loss=criterion(outputs,targets)
            total_loss+=loss

            correct=(outputs.argmax(1)==targets).sum()
            total_correct+=correct
    accuracy=total_correct/len(val_data)
    print(f"验证集的总损失：{total_loss}")
    print(f"预测准确率：{accuracy}")
print("训练结束")
torch.save(myModel.state_dict(),f"myModel_{epoch}.pt")
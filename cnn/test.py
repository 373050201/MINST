import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from model import MyModel
import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt



#加载测试集
print("正在加载测试集...")
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])
test_data=torchvision.datasets.MNIST(root="../dataset",train=False,transform=transform_test)
test_loader=DataLoader(test_data,batch_size=64,drop_last=True,shuffle=True)

myModel=MyModel()
myModel=myModel.cuda()
myModel.load_state_dict(torch.load("./myModel_10.pt"))

print("正在预测...")
total_correct=0
for item in test_loader:
    imgs,targets=item
    imgs=imgs.cuda()
    targets=targets.cuda()

    outputs,_,_,_,_=myModel(imgs)
    correct=(outputs.argmax(1)==targets).sum()
    total_correct += correct
accuracy=total_correct/len(test_data)
print(f"预测准确率：{accuracy}")

show_num=1#随机展示show_num张
for i in range(0,show_num):
    t=random.randint(0,len(test_data)-1)
    img,target=test_data[t]#图片，目标值
    img_np=img.squeeze(0).numpy()#转换为numpy方便展示
    img=img.unsqueeze(0)
    img=img.cuda()
    output,conv1_output,pool1_output,conv2_output,pool2_output=myModel(img)

    #展示中间层输出
    fig,([ax1,ax2],[ax3,ax4])=plt.subplots(2,2,figsize=(8,8))
    #conv1+ReLU
    grid_conv1=make_grid(conv1_output[0].cpu().unsqueeze(1),nrow=8,normalize=True)#排成网格展示
    ax1.imshow(grid_conv1.permute(1,2,0))#CHW转HWC
    ax1.set_title("conv1+ReLU")
    ax1.axis('off')
    #maxPool1
    grid_pool1 = make_grid(pool1_output[0].cpu().unsqueeze(1), nrow=8, normalize=True)  # 排成网格展示
    ax2.imshow(grid_pool1.permute(1,2,0))  # CHW转HWC
    ax2.set_title("max_pool1")
    ax2.axis('off')
    #conv2+ReLU
    grid_conv2=make_grid(conv2_output[0].cpu().unsqueeze(1),nrow=8,normalize=True)
    ax3.imshow(grid_conv2.permute(1,2,0))
    ax3.set_title("conv2+ReLU")
    ax3.axis('off')
    #maxPool2
    grid_pool2=make_grid(pool2_output[0].cpu().unsqueeze(1),nrow=8,normalize=True)
    ax4.imshow(grid_pool2.permute(1,2,0))
    ax4.set_title("max_pool2")
    ax4.axis('off')
    plt.tight_layout()
    plt.show()

    predict=output.argmax(1).item()#预测值
    print(f"当前预测值为{predict}，目标值为{target}")
    plt.figure()
    plt.imshow(img_np,cmap='gray')
    plt.show()
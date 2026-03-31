from torch import nn



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        # self.model1=nn.Sequential(
        #     nn.Conv2d(1,32,3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(32,64,3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2,2),
        #     nn.Flatten(),
        #     nn.Linear(64*7*7,128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.25),
        #     nn.Linear(128,10)
        # )
        #下方即上方代码拆解，便于调试，forward注释同
        self.conv1=nn.Conv2d(1,32,3,padding=1)
        self.relu=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.dropout=nn.Dropout(0.25)
        self.linear1=nn.Linear(64*7*7,128)
        self.linear2=nn.Linear(128,10)
        self.flatten=nn.Flatten()

    def forward(self,x):
        # x = self.model1(x)
        x=self.conv1(x)
        x=self.relu(x)
        conv1_output=x.clone().detach()
        x=self.pool(x)
        pool1_output = x.clone().detach()
        x=self.conv2(x)
        x=self.relu(x)
        conv2_output = x.clone().detach()
        x=self.pool(x)
        pool2_output = x.clone().detach()
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x,conv1_output,pool1_output,conv2_output,pool2_output
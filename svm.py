import random
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import torchvision
import numpy as np

train_data=torchvision.datasets.MNIST(root='./dataset',train=True,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.MNIST(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())

print("正在装载数据...")
sample_size=20000#抽取训练集的样本数量，最大值len(train_data)
x_train=[]
y_train=[]
count=0
for x,y in train_data:
    x_train.append(x)
    y_train.append(y)
    count+=1
    if count==sample_size:
        break
x_test=[]
y_test=[]
for x,y in test_data:
    x_test.append(x)
    y_test.append(y)

def extract_hog(imgs):
    hog_features=[]
    for img in imgs:
        img_np=img.squeeze(0).numpy()
        tmp=hog(img_np,orientations=12,pixels_per_cell=(6,6),cells_per_block=(2,2))
        hog_features.append(tmp)
    return np.array(hog_features)
print("正在提取训练集hog特征...")
x_train_hog=extract_hog(x_train)

print("正在训练svm分类器...")
svm=SVC(kernel='rbf',C=3.0,random_state=42)
svm.fit(x_train_hog,y_train)
# print("正在网格搜索+交叉验证...")
# svm=SVC()
# param_grid={
#     'C':[3,4],
#     'gamma':[0.01,'scale'],
#     'kernel':['rbf']
# }
# grid_search=GridSearchCV(svm,param_grid,cv=3,scoring='accuracy')
# grid_search.fit(x_train_hog,y_train)
# print("最佳参数：",grid_search.best_params_)
# print("最佳得分：",grid_search.best_score_)
# svm=grid_search.best_estimator_

print("正在提取测试集hog特征...")
x_test_hog=extract_hog(x_test)

print("正在预测...")
y_predict=svm.predict(x_test_hog)
accuracy=accuracy_score(y_test,y_predict)

show_num=3#展示随机show_num张
for i in range(0,show_num):
    t=random.randint(0,len(test_data)-1)
    img=x_test[t]
    img_np=img.squeeze(0).numpy()
    predict=y_predict[t]
    target=y_test[t]
    print(f"当前预测值为{predict}，目标值为{target}")
    plt.figure()
    plt.imshow(img_np,cmap='gray')
    plt.show()
print(f"测试集准确率：{accuracy}")
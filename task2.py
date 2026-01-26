import torch
from torch import nn
class Xuecheng_Function_Net(nn.Module):  #创建一个神经网络
    def __init__(self, n_in, hidden, out):
        super().__init__()  #对nn.Module进行初始化

        self.net = nn.Sequential(  #构建神经网络结构
          nn.Linear(n_in, hidden),
          nn.ReLU(),  #使用对回归问题很有用的ReLU激活函数（但是输入端值域限制在0~1）
          nn.Linear(hidden, hidden * 2),
          nn.ReLU(),
          nn.Linear(hidden * 2, hidden * 4),
          nn.ReLU(),
          nn.Linear(hidden * 4, hidden * 2),
          nn.ReLU(),
          nn.Linear(hidden * 2, hidden),
          nn.ReLU(),
          nn.Linear(hidden, out),
        )
    def forward(self, x):   #（前向传播）
        return self.net(x)
def normalize(data,data_max=None,data_min=None): #（构建归一化组件，将数据处理到可以使用ReLU函数）
    if data_max is None:
        data_max = np.max(data)
    if data_min is None:
        data_min = np.min(data)
    data_normalized = (data-data_min)/(data_max-data_min)
    return data_normalized,data_max,data_min
def denormalize(data,data_max,data_min):  #（构建反归一化组件，使模型的预测数据能被还原至原始函数值域来比较）
    return data * (data_max-data_min) + data_min
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
if __name__ == "__main__":
    # 读取CSV文件
    df = pd.read_csv(r'F:\test\task2.csv')
    # 假设列名是'x'和'y'，直接用列名赋值
    x_raw = df.iloc[:,0].values  # 第一列，转为NumPy数组
    y_raw = df.iloc[:,1].values  # 第二列，转为NumPy数组
    x_normalized ,x_max ,x_min = normalize(x_raw)    #归一化组件的使用
    y_normalized ,y_max ,y_min = normalize(y_raw)
    x = x_normalized.reshape(2000,1)
    y = y_normalized.reshape(2000,1)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    model = Xuecheng_Function_Net(n_in=1,hidden=32,out=1)  #使用创建的神经网络进行训练
    criterion = nn.MSELoss()   #计算loss的函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 1000 == 0:  #每1000次训练输出一次loss值
            print(f'After:{epoch}literations, loss:{loss.item()}')
    # 获取预测结果
    h = model(x)
    x_numpy = x.data.numpy()
    h_numpy = h.data.numpy()

    # 将归一化后的数据反归一化到原始范围
    x_original = denormalize(x_numpy, x_max, x_min)
    h_original = denormalize(h_numpy, y_max, y_min)

    # 将一维数组展平，便于绘图
    x_original_flat = x_original.flatten()
    h_original_flat = h_original.flatten()

    # 绘制对比图
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.scatter(x_raw, y_raw, alpha=0.5, s=1, label='raw_data', color='blue')

    # 绘制预测数据
    plt.scatter(x_original_flat, h_original_flat, alpha=0.5, s=1, label='pred_data', color='red')

    # 设置图表属性
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 计算并显示误差统计
    mse = np.mean((y_raw - h_original_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_raw - h_original_flat))


    plt.show()
   
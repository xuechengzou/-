import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条

                                      #1.数据加载部分

def cifar_10_loader(batch_size=64):
    traintransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转，因为测试的图片和训练的图片不一定都是正向摆放的
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(  # 标准化，将图片的RGB值的分布处理的更加均匀，使模型的训练少受亮度等其他的因素影响
            mean=[0.4914, 0.4822, 0.4465],#在网络上查询的训练集图片RGB均值
            std=[0.2023, 0.1994, 0.2010]#在网络上查询到的训练集RGB标准差
        )
    ])
    testtransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=traintransform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=testtransform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"类别: {classes}")

    return trainloader, testloader, classes

                            #定义残差块

class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=2):
            super().__init__()

            # 主路径：两个卷积层
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            # 跳跃连接：如果输入输出尺寸不匹配，需要调整
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            # 主路径
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            # 跳跃连接：加回输入
            out += self.shortcut(x)  # 关键：残差连接
            out = self.relu(out)

            return out

                                     #2.定义ResNet

class My_Simple_Resnet(nn.Module):
    def __init__(self,num_classes=10):    #num_classes是模型需要区分的类别数量
        super(My_Simple_Resnet, self).__init__()

                                     #构建卷积块

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=2, padding=1)  #卷积层，以3x3卷积块，步长为2提取信息
        self.bn1 = nn.BatchNorm2d(64)  #批量归一化层
        self.relu = nn.ReLU(inplace=True)  #激活函数层

                                     #构建残差层



        self.layer1 = self.Residual_layer(64, 64, 2, stride=1)
        self.layer2 = self.Residual_layer(64, 128, 2, stride=2)
        self.layer3 = self.Residual_layer(128, 256, 2, stride=2)
        self.layer4 = self.Residual_layer(256, 512, 2, stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def Residual_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []

            # 第一个块可能需要下采样
            layers.append(ResidualBlock(in_channels, out_channels, stride))

            # 剩余的块保持尺寸不变
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels, stride=1))

            return nn.Sequential(*layers)

    def forward(self, x):
            # 初始卷积
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            # 四个残差层
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            # 分类头
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

                            #4.构建训练函数和评估函数

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'训练 Epoch [{epoch + 1}]')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'Loss': running_loss / (batch_idx + 1),
            'Acc': 100. * correct / total
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy  #输出loss和正确率



def evaluate(model, test_loader, criterion, device):

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

                       #可视化函数

def visualize_training_history(train_losses, train_accs, test_losses, test_accs):
            """可视化训练历史"""
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            epochs = range(1, len(train_losses) + 1)

            # 损失曲线
            axes[0].plot(epochs, train_losses, 'b-', label='训练损失')
            axes[0].plot(epochs, test_losses, 'r-', label='测试损失')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('损失')
            axes[0].set_title('训练和测试损失')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 准确率曲线
            axes[1].plot(epochs, train_accs, 'b-', label='训练准确率')
            axes[1].plot(epochs, test_accs, 'r-', label='测试准确率')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('准确率 (%)')
            axes[1].set_title('训练和测试准确率')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
            plt.show()

def visualize_predictions(model, test_loader, classes, device, num_images=16):
                """可视化模型预测结果"""
                model.eval()

                # 获取一批测试数据
                data_iter = iter(test_loader)
                images, labels = next(data_iter)
                images, labels = images[:num_images], labels[:num_images]

                # 预测
                with torch.no_grad():
                    images_gpu = images.to(device)
                    outputs = model(images_gpu)
                    _, predictions = outputs.max(1)

                # 将图像反标准化以便显示
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                images = images * std + mean
                images = torch.clamp(images, 0, 1)

                # 绘制图像和预测
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                axes = axes.ravel()

                for i in range(num_images):
                    # 转换图像格式: (C, H, W) -> (H, W, C)
                    img = images[i].permute(1, 2, 0).numpy()

                    true_label = classes[labels[i]]
                    pred_label = classes[predictions[i]]

                    # 设置颜色：正确预测为绿色，错误为红色
                    color = 'green' if labels[i] == predictions[i] else 'red'

                    axes[i].imshow(img)
                    axes[i].set_title(f"真值: {true_label}\n预测: {pred_label}", color=color, fontsize=10)
                    axes[i].axis('off')

                plt.suptitle("模型预测结果", fontsize=16)
                plt.tight_layout()
                plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
                plt.show()

                       #       主训练函数

def main():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置超参数
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.1

    print("=" * 60)
    print("My_Simple_ResNet CIFAR-10图像分类")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载CIFAR-10数据集...")
    train_loader, test_loader, classes = cifar_10_loader(batch_size=batch_size)

    # 2. 创建模型
    print("\n2. 创建My_Simple_ResNet模型...")
    model = My_Simple_Resnet(num_classes=10).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    # 4. 训练模型
    print("\n3. 开始训练...")
    print(f"  训练轮数: {num_epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  初始学习率: {learning_rate}")
    print("-" * 60)

    # 记录训练历史
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 记录结果
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, 'best_model.pth')

        # 打印epoch结果
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
                  f"LR: {learning_rate:.4f} | "
                  f"训练损失: {train_loss:.4f} | "
                  f"训练准确率: {train_acc:6.2f}% | "
                  f"测试损失: {test_loss:.4f} | "
                  f"测试准确率: {test_acc:6.2f}%")

    print("-" * 60)
    print(f"训练完成!")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")

    # 5. 可视化训练历史
    print("\n4. 可视化训练历史...")
    visualize_training_history(train_losses, train_accuracies, test_losses, test_accuracies)

    # 6. 可视化预测结果
    print("\n5. 可视化预测结果...")
    visualize_predictions(model, test_loader, classes, device)

    # 7. 评估每个类别的准确率
    print("\n6. 按类别评估准确率...")
    model.eval()
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()

            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 打印每个类别的准确率
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'  类别 {classes[i]:10s}: {accuracy:5.2f}%')

    # 计算整体准确率
    total_accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f"  整体准确率: {total_accuracy:.2f}%")

    return best_accuracy


# ============================================
# 运行主函数
# ============================================
if __name__ == "__main__":
    try:
        print("开始训练My_Simple_ResNet...")
        print("=" * 60)

        best_acc = main()
        print(f"\n训练完成！最终最佳准确率 = {best_acc:.2f}%")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
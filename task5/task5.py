import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime
import random
                              #解决中文字体显示问题
matplotlib.rcParams.update({
    'font.family': 'SimHei',  # 主字体
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'],  # 备用字体列表
    'axes.unicode_minus': False,  # 解决负号显示问题
    'text.usetex' : False
})
                              #设置随机种子，确保复现一致性

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

                              #U-Net主体构建

class DoubleConv(nn.Module):                                                #双重卷积核模块
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(                                   #创建双重卷积核模块容器
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):                                                       #下采样模块
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_cov = nn.Sequential(                                    #最大池化确保边界特征分明
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_cov(x)



class Up(nn.Module):                                                         #上采样模块
    def __init__(self, in_channels, out_channels):
        super().__init__()
                                                                             #使用双线性插值进行图像扩充
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1,x2):                                                #x1为解码器直接接收图片，x2为跳跃连接图片
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]                                  #计算图片尺寸差异
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2 , diffY // 2, diffY - diffY // 2])
        # 跳跃连接：拼接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):                                                    #输出层
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        """
        参数:
            n_channels: 输入图像的通道数 (RGB=3, 灰度=1)
            n_classes: 分割类别数，二分类问题设为1
            bilinear: 是否使用双线性插值进行上采样
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes


        # 编码器部分
        self.inc = DoubleConv(n_channels, 64)  # 输入 -> 64
        self.down1 = Down(64, 128)  # 64 -> 128
        self.down2 = Down(128, 256)  # 128 -> 256
        self.down3 = Down(256, 512)  # 256 -> 512

        # 中间部分（最底层）
        factor = 2
        self.down4 = Down(512, 1024 // factor)  # 512 -> 1024

        # 解码器部分
        self.up1 = Up(1024, 512 // factor)  # 1024 -> 512
        self.up2 = Up(512, 256 // factor)  # 512 -> 256
        self.up3 = Up(256, 128 // factor)  # 256 -> 128
        self.up4 = Up(128, 64)  # 128 -> 64

        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 1024, H/16, W/16]

        # 解码器（带跳跃连接）
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)  # [B, 256, H/4, W/4]
        x = self.up3(x, x2)  # [B, 128, H/2, W/2]
        x = self.up4(x, x1)  # [B, 64, H, W]

        # 输出
        logits = self.outc(x)  # [B, n_classes, H, W]
        return logits

                                   #数据集加载部分

class PennFudanDataset(Dataset):

    def __init__(self, root_dir, transform=None, img_size=256):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_size = img_size

        # 获取所有图像文件
        self.image_dir = self.root_dir / 'PNGImages'
        self.mask_dir = self.root_dir / 'PedMasks'

        # 获取图像文件名列表
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        self.mask_files = sorted(list(self.mask_dir.glob('*.png')))

        # 确保图像和掩码数量一致
        assert len(self.image_files) == len(self.mask_files), \
            "图像和掩码数量不匹配"

        print(f"加载数据集: {len(self.image_files)} 个样本")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # 将掩码转换为二值掩码（行人=1，背景=0）
        # 原数据集中背景=0，行人实例=1,2,3...
        mask_np = np.array(mask)
        binary_mask = np.where(mask_np > 0, 1, 0).astype(np.float32)
        binary_mask = Image.fromarray(binary_mask)

        # 应用数据增强
        if self.transform:
            # 对图像应用变换
            image = self.transform(image)

            # 对掩码应用相同的几何变换
            # 注意：掩码不需要归一化
            mask_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
            binary_mask = mask_transform(binary_mask)

        return image, binary_mask

                                        #损失函数和评估指标

class SegmentationLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # 展平
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # 计算交集和并集
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
               (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)

        # 组合损失
        total_loss = bce + dice

        return total_loss, bce, dice


def calculate_metrics(predictions, targets, threshold=0.5):

    # 应用sigmoid并二值化
    preds_binary = (torch.sigmoid(predictions) > threshold).float()

    # 计算TP, FP, TN, FN
    tp = ((preds_binary == 1) & (targets == 1)).sum().item()          #预测为正实际也为正
    fp = ((preds_binary == 1) & (targets == 0)).sum().item()          #预测为正但实际为负
    tn = ((preds_binary == 0) & (targets == 0)).sum().item()          #预测为正实际也为正
    fn = ((preds_binary == 0) & (targets == 1)).sum().item()          #预测为负但实际为正

    # 避免除以0
    epsilon = 1e-7

    # 计算指标
    accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

                                    #U-Net网络主体

class UNetTrainer:

    def __init__(self, model, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = SegmentationLoss()

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': []
        }

        # 创建保存目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path('checkpoints') / f'unet_pennfudan_{timestamp}'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"模型保存目录: {self.save_dir}")

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        metrics_list = []

        pbar = tqdm(train_loader, desc='训练', leave=False)
        for images, masks in pbar:
            # 移动数据到设备
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 计算损失
            total_loss, bce_loss, dice_loss = self.criterion(outputs, masks)

            # 反向传播
            total_loss.backward()
            self.optimizer.step()

            # 记录损失
            epoch_loss += total_loss.item()
            epoch_dice += dice_loss.item()

            # 计算指标
            with torch.no_grad():
                metrics = calculate_metrics(outputs, masks)
                metrics_list.append(metrics)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Dice': f'{dice_loss.item():.4f}'
            })

        # 计算平均指标
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)

        # 计算平均指标
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])

        return avg_loss, avg_dice, avg_metrics

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        metrics_list = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='验证', leave=False)
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                total_loss, bce_loss, dice_loss = self.criterion(outputs, masks)

                val_loss += total_loss.item()
                val_dice += dice_loss.item()

                # 计算指标
                metrics = calculate_metrics(outputs, masks)
                metrics_list.append(metrics)

                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice_loss.item():.4f}'
                })

        # 计算平均指标
        avg_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)

        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])

        return avg_loss, avg_dice, avg_metrics

    def train(self, train_loader, val_loader, num_epochs=30):
        """完整训练循环"""
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # 训练
            start_time = time.time()
            train_loss, train_dice, train_metrics = self.train_epoch(train_loader)
            val_loss, val_dice, val_metrics = self.validate(val_loader)
            epoch_time = time.time() - start_time

            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(current_lr)

            # 打印结果
            print(f"时间: {epoch_time:.1f}s, LR: {current_lr:.2e}")
            print(f"训练 Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"验证 Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            print(f"训练 IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"验证 IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f'best_model_epoch{epoch + 1}.pth')
                print(f"保存最佳模型 (验证损失: {val_loss:.4f})")

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch + 1}.pth')
                self.save_history()

        # 保存最终模型和历史
        self.save_checkpoint('final_model.pth')
        self.save_history()
        print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
        print(f"检查点已保存: {self.save_dir / filename}")

    def save_history(self):
        """保存训练历史"""
        # 保存为JSON
        json_path = self.save_dir / 'training_history.json'
        json_history = {}
        for key, value in self.history.items():
            json_history[key] = [float(v) for v in value]

        with open(json_path, 'w') as f:
            json.dump(json_history, f, indent=2)

        print(f"训练历史已保存: {json_path}")

        # 绘制并保存图表
        self.plot_training_history()

    def plot_training_history(self):
        """绘制训练历史图表"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.history['val_loss'], label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Dice系数曲线
        axes[0, 1].plot(self.history['train_dice'], label='训练Dice')
        axes[0, 1].plot(self.history['val_dice'], label='验证Dice')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Loss')
        axes[0, 1].set_title('Dice损失曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # IoU曲线
        axes[0, 2].plot(self.history['train_iou'], label='训练IoU')
        axes[0, 2].plot(self.history['val_iou'], label='验证IoU')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].set_title('IoU曲线')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # F1分数曲线
        axes[1, 0].plot(self.history['train_f1'], label='训练F1')
        axes[1, 0].plot(self.history['val_f1'], label='验证F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1分数曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 学习率曲线
        axes[1, 1].plot(self.history['learning_rate'], label='学习率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 精度-召回曲线
        axes[1, 2].plot(self.history['train_f1'], label='F1')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('综合指标')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        history_plot_path = self.save_dir / 'training_history.png'
        plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图形以避免在控制台显示
        print(f"训练历史图表已保存: {history_plot_path}")

    def visualize_predictions(self, dataloader, num_samples=3):
        """可视化预测结果"""
        self.model.eval()

        # 获取一个批次的数据
        images, masks = next(iter(dataloader))
        images = images[:num_samples].to(self.device)
        masks = masks[:num_samples].to(self.device)

        with torch.no_grad():
            predictions = self.model(images)
            pred_masks = torch.sigmoid(predictions) > 0.5

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 原始图像
            img_np = images[i].cpu().permute(1, 2, 0).numpy()
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f'样本 {i + 1}: 原始图像')
            axes[i, 0].axis('off')

            # 真实掩码
            axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
            axes[i, 1].set_title('真实分割')
            axes[i, 1].axis('off')

            # 预测掩码
            axes[i, 2].imshow(pred_masks[i].cpu().squeeze(), cmap='gray')
            axes[i, 2].set_title('预测分割')
            axes[i, 2].axis('off')

            # 叠加效果
            axes[i, 3].imshow(img_np)
            axes[i, 3].imshow(pred_masks[i].cpu().squeeze(), cmap='jet', alpha=0.5)
            axes[i, 3].set_title('分割结果叠加')
            axes[i, 3].axis('off')

        plt.tight_layout()
        pred_plot_path = self.save_dir / 'predictions_visualization.png'
        plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"预测可视化已保存: {pred_plot_path}")

                                      #主程序

def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据集路径
    # 注意：需要先下载PennFudanPed数据集
    # 下载地址: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    dataset_path = 'PennFudanPed'  # 修改为你的数据集路径

    if not Path(dataset_path).exists():
        print(f"\n错误: 数据集路径不存在: {dataset_path}")
        print("请按以下步骤下载数据集:")
        print("1. 下载链接: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip")
        print("2. 解压到当前目录")
        print("3. 确保目录结构如下:")
        print("   PennFudanPed/")
        print("   ├── PNGImages/")
        print("   └── PedMasks/")
        return

    # 定义数据变换
    img_size = 256
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    print("\n创建数据集...")
    full_dataset = PennFudanDataset(dataset_path, transform=None, img_size=img_size)

    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 为子数据集设置不同的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 可视化一个样本
    print("\n数据集样本展示...")
    sample_idx = 0
    image, mask = train_dataset[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 显示图像（反归一化）
    img_np = image.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    axes[0].imshow(img_np)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 显示掩码
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('分割掩码')
    axes[1].axis('off')

    # 显示叠加效果
    axes[2].imshow(img_np)
    axes[2].imshow(mask.squeeze(), cmap='jet', alpha=0.5)
    axes[2].set_title('分割结果叠加')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 创建数据加载器
    batch_size = 4
    num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 初始化模型
    print("\n初始化U-Net模型...")
    model = UNet(n_channels=3, n_classes=1)

    # 创建训练器
    trainer = UNetTrainer(model, device=device, learning_rate=1e-4)

    # 开始训练
    num_epochs = 30
    print(f"\n开始训练，共{num_epochs}个epoch...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # 可视化预测结果
    print("\n预测结果可视化...")
    trainer.visualize_predictions(val_loader, num_samples=3)

    # 评估最终模型
    print("\n最终模型评估:")
    trainer.model.eval()
    with torch.no_grad():
        val_loss, val_dice, val_metrics = trainer.validate(val_loader)
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证Dice: {val_dice:.4f}")
        print(f"验证IoU: {val_metrics['iou']:.4f}")
        print(f"验证F1分数: {val_metrics['f1']:.4f}")
        print(f"验证准确率: {val_metrics['accuracy']:.4f}")
        print(f"验证精确率: {val_metrics['precision']:.4f}")
        print(f"验证召回率: {val_metrics['recall']:.4f}")

    print(f"\n训练完成！所有文件保存在: {trainer.save_dir}")
    return trainer

                                     #推理函数

def predict_single_image(model_path, image_path, device='cpu'):
    """
    对单张图像进行预测

    参数:
        model_path: 模型路径
        image_path: 图像路径
        device: 设备
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output) > 0.5

    # 将掩码调整回原始大小
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    pred_mask_resized = Image.fromarray(pred_mask_np.astype(np.uint8) * 255)
    pred_mask_resized = pred_mask_resized.resize(original_size, Image.NEAREST)

    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(pred_mask_resized, cmap='gray')
    axes[1].set_title('预测掩码')
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(pred_mask_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('分割结果叠加')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('single_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

    return pred_mask_resized


if __name__ == "__main__":
    print("=" * 60)
    print("U-Net在PennFudanPed数据集上的语义分割")
    print("=" * 60)

    # 检查是否有GPU
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("未检测到GPU，使用CPU训练（可能较慢）")

    # 运行主程序
    trainer = main()

    print("\n" + "=" * 60)
    print("使用训练好的模型进行推理:")
    print("=" * 60)
    print("调用 predict_single_image() 函数对单张图像进行预测")

    print("示例: predict_single_image('checkpoints/.../best_model.pth', 'test_image.png')")

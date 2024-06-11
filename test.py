import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from model import ModifiedLeNet5


# 自定义数据集类
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, -1])
        image = read_image(img_path).float() / 255.0  # 将图像像素值归一化到0-1之间
        label = torch.tensor(int(self.annotations.iloc[idx, -2]))

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 设置数据集路径
    train_csv = 'GTSRB/train.csv'
    test_csv = 'GTSRB/test.csv'
    root_dir = 'GTSRB'

    # 加载数据集
    train_dataset = GTSRBDataset(csv_file=train_csv, root_dir=root_dir, transform=transform)
    test_dataset = GTSRBDataset(csv_file=test_csv, root_dir=root_dir, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 模型实例化
    model = ModifiedLeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    num_epochs = 20
    best_val_accuracy = 0.0
    best_model_path = 'best_modified_lenet5.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with accuracy: {best_val_accuracy:.2f}%')

    print(f'Training complete. Best validation accuracy: {best_val_accuracy:.2f}%')

if __name__ == '__main__':
    main()


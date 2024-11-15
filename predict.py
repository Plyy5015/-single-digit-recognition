import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 以下是你的原代码部分
import torch
from model import AlexNet
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# 继续原来的代码
def load_image(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((227, 227))
        image = Image.fromarray(255 - np.array(image))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        print(f"无法加载图片：{e}")
        return None

def predict(model, image, device):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    return prediction.item()

def test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = 'Number.jpg'  # 本地图片路径
    model = AlexNet(num_classes=10).to(device)
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    print("模型权重已加载")
    image = load_image(image_path)
    if image is None:
        return
    prediction = predict(model, image, device)
    print(f"预测标签: {prediction}")
    transform = transforms.Compose([transforms.Resize(227), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    accuracy = test_accuracy(model, test_loader, device)
    print(f"模型在测试集上的准确度: {accuracy:.2f}%")

    fig, ax = plt.subplots()
    ax.imshow(image.cpu().squeeze(), cmap='gray')
    ax.axis('off')
    plt.title(f"预测: {prediction} - 准确度: {accuracy:.2f}%", fontsize=14, color='white')
    plt.show()

if __name__ == "__main__":
    main()

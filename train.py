# train.py
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from model import AlexNet
from data_processing import load_data

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Training Loss: {loss.item():.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001

    train_loader, _ = load_data(batch_size)
    model = AlexNet(num_classes=10).to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)

    # 保存模型权重
    torch.save(model.state_dict(), 'model_weights.pth')
    print("模型已保存为 model_weights.pth")

if __name__ == "__main__":
    main()

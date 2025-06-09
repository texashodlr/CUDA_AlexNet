import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

"""
This is a single file pytorch implementation of Alex Net!
1. There's five convolutional layers (96/256/384/384/256 kernels)
2. There's ReLU nonlinearity after each conv layer and FCL
3. Local response normalziation after the first and second conv layers
4. Overlapping max-pooling -- 3x3 kernel, stride 2--after the first second and fifth conv
5. Three FC layers (4096, 4096, 1000 neurons) with dropout = 0.5 in the first two
CIFAR-10 Link: https://www.cs.toronto.edu/~kriz/
"""

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
                # Conv Layer #1
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2)
                # Conv Layer #2
                nn.Conv2d(96, 256,kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # Conv Layer #3
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                #Conv Layer #4
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                #Conv Layer #5
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256*6*6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

# Function to compute top-1 and top-5 accuracy
def compute_accuracy(output, target, topk=1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


# Training Function
def train(model, train_loaderm criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = mode(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99: # Print every 100 batches
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

# Evaluation Function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion (outputs, targets)
            running_loss += loss.item()
            acc1, acc5 = compute_accuracy(outputs, targets, topk=(1, 5))
            top1_acc += acc1.item() * inputs.size(0)
            top5_acc += acc5.item() * inputs.size(0)
            total += inputs.size(0)
    avg_loss = running_loss / len(val_loader)
    avg_top1 = top1_acc / total
    avg_top5 = top5_acc / total
    return avg_loss, avg_top1, avg_top5

# Main function
def main():
    # Hyperparameters from the Alex Net Paper
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    num_epochs = 90      # Paper trained for 90ish cycles
    num_classes = 1000   # For ImageNet
    use_cifar10 = True   # Set to False for ImageNet

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device")

    # Data Augmentation and preprocessing
    if use_cifar10:
        # CIFAR-10 preprocessing (32x32 images, resize to 224x224 for AlexNet)
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        train_dataset = datasets.CIFAR10(root='../data/cifar-10-batches-py/', train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='../data/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        # ImageNet preprocessing (assumes data in imagenet_dir)
        imagenet_dir = '/path/to/imagenet'  # Update this path
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(os.path.join(imagenet_dir, 'train'), transform=transform_train)
        val_dataset = datasets.ImageFolder(os.path.join(imagenet_dir, 'val'), transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler (reduce LR by factor of 10 three times)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
            print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.3f}, '
                  f'Top-1 Acc: {val_top1:.2f}%, Top-5 Acc: {val_top5:.2f}%')

    # Final evaluation
    val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
    print(f'Final: Val Loss: {val_loss:.3f}, Top-1 Acc: {val_top1:.2f}%, Top-5 Acc: {val_top5:.2f}%')

if __name__ == '__main__':
    main()



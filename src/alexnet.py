import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2)
                nn.Conv2d(96, 256,kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
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



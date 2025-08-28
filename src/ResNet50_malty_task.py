import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import time, os
from tempfile import TemporaryDirectory

# Paths
data_dir = '/Users/chootydoony/Documents/Miun/EL035A_Project/Project-II/Thin_wedge/Dataset_4'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Normalization: Custom
mean = [0.9892, 0.9934, 0.9901]
std = [0.0884, 0.0590, 0.0779]

# Transforms
transforms_train = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transforms_val = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Data
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms_val),
}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False),
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Model
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(class_names))
)
model = model.to(device)

# Optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0005, weight_decay=0.0005)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    since = time.time()
    best_acc = 0.0
    epochs_no_improve = 0

    with TemporaryDirectory() as tempdir:
        best_model_path = os.path.join(tempdir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_path)
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if epoch >= 15 and epochs_no_improve >= patience:
                print("Early stopping.")
                break

        model.load_state_dict(torch.load(best_model_path))
        print(f'Training complete. Best val Acc: {best_acc:.4f}')
        return model

# Evaluation
def evaluate_model(model):
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_val)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names))

# Run
model = train_model(model, criterion, optimizer, scheduler)
evaluate_model(model)

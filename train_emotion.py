import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ========================================================
#  CUSTOMIZED RESNET18 for 1-channel input (Grayscale)
# ========================================================
class ResNet18_Gray(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet18(weights=None)    # train from scratch
        # Change conv1 from 3→1 channel
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ========================================================
#  EVALUATE
# ========================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)

            total_loss += loss.item()
            preds = output.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), (100 * correct / total)


# ========================================================
# TRAIN
# ========================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                train_data, SAVE_PATH, EPOCHS, PATIENCE, device):

    best_acc = 0
    patience_counter = 0

    print("\n===== START TRAINING (GRAYSCALE-1CH) =====")

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = 100 * correct / len(train_data)
        train_loss = running_loss / len(train_loader)

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 Saved best model at epoch {epoch+1} (Acc={best_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("⛔ EARLY STOPPING")
            break

    print(f"\n🎉 Training complete. Best Accuracy = {best_acc:.2f}%")
    return best_acc


# ========================================================
# CONFUSION MATRIX
# ========================================================
def test_confusion(model, test_loader, class_names, SAVE_PATH, device):
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_true, y_pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=class_names))


# ========================================================
# MAIN
# ========================================================
def main():

    DATASET_DIR = "./dataset"
    SAVE_PATH = "./models/resnet18_gray.pth"

    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 80
    PATIENCE = 10
    NUM_WORKERS = 0    # Windows spawn fix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # -------- GRAYSCALE TRANSFORM --------
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # -------- LOAD DATA --------
    train_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tf)
    test_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=test_tf)

    class_names = train_data.classes
    print("Classes:", class_names)

    # -------- STRONG IMBALANCE FIX --------
    labels = [label for _, label in train_data]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.DoubleTensor([class_weights[l] for l in labels])

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    # -------- MODEL: RESNET18(1 CH) --------
    model = ResNet18_Gray(num_classes=len(class_names)).to(device)

    # -------- CLASS-WEIGHT LOSS: chống imbalance Level 2 --------
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # -------- TRAIN --------
    best_acc = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, scheduler,
        train_data, SAVE_PATH, EPOCHS, PATIENCE, device
    )

    # -------- EVALUATE --------
    test_confusion(model, test_loader, class_names, SAVE_PATH, device)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

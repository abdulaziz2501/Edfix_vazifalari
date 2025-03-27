import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Qurilma (device) tanlash
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Foydalanilayotgan qurilma: {device}")

# Ma'lumotlarni o'zgartirish (transformatsiya)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ma'lumotlar to'plamini yuklash
def load_data(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, 'training/')
    val_dir = os.path.join(data_dir, 'PDR/')
    
    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = ImageFolder(root=val_dir, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.classes

# Model arxitekturasi
class MangaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MangaClassifier, self).__init__()
        # ResNet50 ni o'rnatilgan vazn (weight)lar bilan yuklash
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        # So'nggi to'liq ulanish (fully connected) qatlamni almashtirish
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# O'rgatish funksiyasi
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, scheduler=None):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if scheduler:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validatsiya
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        # Eng yaxshi modelni saqlash
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_manga_classifier.pth')
            print(f'Model saqlandi! Yangi eng yaxshi aniqlik: {best_acc:.4f}')
    
    return model, train_losses, val_losses, train_accs, val_accs

# Modelni baholash funksiyasi
def evaluate_model(model, data_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Aniqlikni hisoblash
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Aniqlik: {accuracy:.4f}')
    
    # Confusion matrix chizish
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Bashorat qilingan')
    plt.ylabel('Haqiqiy')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Batafsil hisobot
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    return accuracy, cm, report

# Asosiy funksiya
def main(data_dir, num_epochs=15, batch_size=32, learning_rate=0.001):
    # Ma'lumotlarni yuklash
    train_loader, val_loader, class_names = load_data(data_dir, batch_size)
    num_classes = len(class_names)
    print(f'Sinflar soni: {num_classes}')
    print(f'Sinflar: {class_names}')
    
    # Modelni yaratish
    model = MangaClassifier(num_classes)
    model = model.to(device)
    
    # Loss funksiyasi va optimizatorni tanlash
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Modelni o'rgatish
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler
    )
    
    # Eng yaxshi modelni yuklash
    model.load_state_dict(torch.load('best_manga_classifier.pth'))
    
    # Modelni baholash
    accuracy, cm, report = evaluate_model(model, val_loader, class_names)
    
    # O'rgatish jarayoni grafigini chizish
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_curves.png')
    
    return model, class_names, accuracy

# Yangi rasmni bashorat qilish funksiyasi
def predict_image(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted_idx = torch.max(output, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()

# Misol uchun ishlatish
if __name__ == "__main__":
    # Ma'lumotlar papkasi
    data_dir = "diabetic/"  # Bu yerga o'z papkangiz yo'lini kiriting
    
    # Modelni o'rgatish
    model, class_names, accuracy = main(data_dir, num_epochs=15, batch_size=32)
    
    # Yangi rasmni bashorat qilish
    image_path = "diabetic/PDR/570.jpg"  # Bu yerga test rasm yo'lini kiriting
    predicted_class, confidence, all_probs = predict_image(model, image_path, class_names)
    
    print(f"Bashorat qilingan sinf: {predicted_class}")
    print(f"Ishonch: {confidence:.4f}")
    
    # Barcha ehtimolliklarni ko'rsatish
    for i, prob in enumerate(all_probs):
        print(f"{class_names[i]}: {prob:.4f}")
    
    # Rasmni ko'rsatish
    plt.figure(figsize=(8, 6))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Bashorat: {predicted_class} ({confidence:.4f})")
    plt.axis('off')
    plt.savefig('prediction_result.png')

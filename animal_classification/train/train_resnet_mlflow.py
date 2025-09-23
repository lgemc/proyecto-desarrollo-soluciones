#!/usr/bin/env python3
"""
Script de entrenamiento de ResNet50 con MLFlow para clasificación de imágenes
"""

import argparse
import torch
from torchvision import transforms, datasets
from torch import nn
from torch import optim
import numpy as np
from sklearn.metrics import classification_report
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
import mlflow
from pathlib import Path

def setup_device():
    """Configura el dispositivo de procesamiento"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Use Apple Metal
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
    else:
        device = torch.device("cpu")   # Fallback to CPU
    
    print(f"Using device: {device}")
    return device

def get_transforms():
    """Define las transformaciones de preprocesamiento"""
    preprocess = transforms.Compose([
        transforms.Resize(256),           # Redimensiona manteniendo proporción
        transforms.CenterCrop(224),       # Recorte central a 224x224
        transforms.ToTensor(),            # Convierte a tensor PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # Normalización ImageNet
    ])
    return preprocess

def load_data(data_path, batch_size=32, train_split=0.8):
    """Carga y prepara los datos"""
    preprocess = get_transforms()
    
    dataset = datasets.ImageFolder(root=data_path, transform=preprocess)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, dataset.classes

class FinetuneResnet(nn.Module):
    """Modelo ResNet50 con fine-tuning para clasificación"""
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_feats = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, num_classes),
        )
        
        # Congelar backbone, solo entrenar clasificador
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.fc.parameters():
            p.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)  # retorna logits

def evaluate_model(model, test_loader, criterion, device):
    """Evalúa el modelo con datos de prueba"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    return test_loss, accuracy

def detailed_evaluation(model, test_loader, class_names, device):
    """Evaluación detallada con reporte de clasificación y matriz de confusión"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    print(report)

    return all_predictions, all_targets, all_probabilities, report


def calculate_per_class_accuracy(predictions, targets, class_names):
    """Calcular la exactitud para cada clase"""
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = (np.array(targets) == i)
        if class_mask.sum() > 0:
            class_predictions = np.array(predictions)[class_mask]
            class_targets = np.array(targets)[class_mask]
            accuracy = (class_predictions == class_targets).mean() * 100
            per_class_acc[class_name] = accuracy
        else:
            per_class_acc[class_name] = 0.0

    print("\nExactitud por clase:")
    for class_name, acc in per_class_acc.items():
        print(f"{class_name}: {acc:.2f}%")

    return per_class_acc

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                device, num_epochs=10):
    """Entrenar el modelo y registrar métricas"""
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            #if batch_idx % 10 == 0:
            #    print(f'Época {epoch + 1}/{num_epochs}, Lote {batch_idx}/{len(train_loader)}, Pérdida: {loss.item():.4f}')

        # Calcular métricas de entrenamiento
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Fase de evaluación
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Logging a MLFlow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_accuracy", test_acc, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

        # Guardar el mejor modelo
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()

        # Actualizar la tasa de aprendizaje
        scheduler.step()

        print(f'Época {epoch + 1}/{num_epochs}:')
        print(f'Pérdida Entrenamiento: {train_loss:.4f}, Exactitud Entrenamiento: {train_acc:.2f}%')
        print(f'Pérdida Prueba: {test_loss:.4f}, Exactitud Prueba: {test_acc:.2f}%')
        print(f'Tasa de Aprendizaje: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)

    # Cargar el mejor modelo
    model.load_state_dict(best_model_state)
    print(f'Mejor Exactitud en Prueba: {best_test_acc:.2f}%')

    return train_losses, train_accuracies, test_losses, test_accuracies, best_test_acc

def main():
    parser = argparse.ArgumentParser(description='Entrenar ResNet50 con MLFlow')
    parser.add_argument('--data_path', type=str, default='data', 
                        help='Ruta a los datos de entrenamiento')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Tamaño del lote')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                        help='Tasa de aprendizaje')
    parser.add_argument('--experiment_name', type=str, default='resnet50_classification',
                        help='Nombre del experimento en MLFlow')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Nombre del run en MLFlow')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Tasa de dropout')
    args = parser.parse_args()

    #mlflow.set_tracking_uri("http://54.198.195.213:8050")
    # Configurar MLFlow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        # Configurar dispositivo
        device = setup_device()
        
        # Cargar datos
        print("Cargando datos...")
        train_loader, test_loader, class_names = load_data(
            args.data_path, args.batch_size
        )
        
        num_classes = len(class_names)
        print(f"Clases encontradas: {class_names}")
        print(f"Número de clases: {num_classes}")
        
        # Log de parámetros
        mlflow.log_param("device", str(device))
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("class_names", class_names)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("train_split", 0.8)
        mlflow.log_param("model_architecture", "ResNet50")
        mlflow.log_param("pretrained_weights", "IMAGENET1K_V2")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("scheduler", "StepLR")
        mlflow.log_param("scheduler_step_size", 5)
        mlflow.log_param("scheduler_gamma", 0.1)
        mlflow.log_param("dropout_rate", args.dropout_rate)
        
        # Inicializar modelo
        model = FinetuneResnet(num_classes, args.dropout_rate)
        model = model.to(device)
        
        # Función de pérdida y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lr=args.learning_rate, params=model.parameters())
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Entrenar modelo
        print("Iniciando entrenamiento...")
        train_losses, train_accs, test_losses, test_accs, best_test_acc = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, 
            device, args.epochs
        )
        
        # Log métricas finales
        mlflow.log_metric("final_best_accuracy", best_test_acc)
        
        
        # Evaluación detallada
        #print("Evaluación detallada...")
        predictions, targets, _, report = detailed_evaluation(
            model, test_loader, class_names, device
        )
        
        # Calcular exactitud por clase
        per_class_acc = calculate_per_class_accuracy(predictions, targets, class_names)
        
        # Log métricas por clase
        for class_name, acc in per_class_acc.items():
            mlflow.log_metric(f"accuracy_{class_name}", acc)
        
        model_path = Path("models/animal-classifier-resnet.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_names': class_names,
            'best_test_accuracy': best_test_acc
        }, model_path)
        
        print(f"Modelo guardado en: {model_path}")
        
        # Log del modelo en MLFlow
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name=f"resnet50_classifier_{args.experiment_name}"
        )
        
        # Log artefacto del modelo
        mlflow.log_artifact(str(model_path))
        
        print(f"Experimento completado. Mejor exactitud: {best_test_acc:.2f}%")
 

if __name__ == "__main__":
    main()

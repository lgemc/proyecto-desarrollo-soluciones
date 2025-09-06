#!/usr/bin/env python3
"""
Script de entrenamiento de Vision Transformer (ViT) con MLFlow para clasificación de imágenes
Migrado desde notebook transformer-encoder-classifier.ipynb
"""


import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report
from torchvision import datasets
from transformers import ViTImageProcessorFast, ViTForImageClassification
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
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

def vit_transform(processor):
    """Crea función de transformación usando ViT processor"""
    def transform_fn(image):
        """Transform function that processes PIL images using ViT processor."""
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension
    return transform_fn

def load_data(data_path, processor, batch_size=32, train_split=0.7, val_split=0.15):
    """Carga y prepara los datos con ViT processor"""
    
    # Create dataset with ViT transform
    transform_fn = vit_transform(processor)
    dataset = datasets.ImageFolder(root=data_path, transform=transform_fn)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Collate function for ViT
    def collate_fn(batch):
        pixel_values = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, dataset.classes

def create_vit_model(model_name, num_labels, class_names):
    """Crea modelo ViT para clasificación"""
    
    # Create label mappings
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}
    
    # Load model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    return model

def evaluate_model(model, data_loader, device):
    """Evalúa el modelo con datos de prueba"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels_batch = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            total_correct += (predictions == labels_batch).sum().item()
            total_samples += labels_batch.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy

def detailed_evaluation(model, test_loader, class_names, device):
    """Evaluación detallada con reporte de clasificación y matriz de confusión"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels_batch = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels_batch.cpu().numpy())
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

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3):
    """Entrenar el modelo y registrar métricas"""
    
    best_val_accuracy = 0.0
    train_losses = []
    val_accuracies = []

    print("Starting fine-tuning process...")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels_batch = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels_batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            current_avg_loss = total_train_loss / (train_pbar.n + 1)
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{current_avg_loss:.4f}'
            })

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- VALIDATION PHASE ---
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        # Logging a MLFlow
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        
        # Print epoch summary
        print(f"\n Epoch {epoch + 1} Summary:")
        print(f"   Training Loss: {avg_train_loss:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best accuracy! Saving model...")
        else:
            print(f"No improvement. Best remains: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Fine-tuning completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"{'='*60}")

    return train_losses, val_accuracies, best_val_accuracy

def main():
    parser = argparse.ArgumentParser(description='Entrenar Vision Transformer con MLFlow')
    parser.add_argument('--data_path', type=str, default='data', 
                        help='Ruta a los datos de entrenamiento')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Tamaño del lote')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                        help='Tasa de aprendizaje')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224',
                        help='Nombre del modelo ViT preentrenado')
    parser.add_argument('--experiment_name', type=str, default='vit_classification',
                        help='Nombre del experimento en MLFlow')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Nombre del run en MLFlow')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Proporción para entrenamiento')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Proporción para validación')
    
    args = parser.parse_args()

    #mlflow.set_tracking_uri("http://54.198.195.213:8050")
    # Configurar MLFlow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        # Configurar dispositivo
        device = setup_device()
        
        # Cargar processor
        print("Cargando ViT processor...")
        processor = ViTImageProcessorFast.from_pretrained(args.model_name)
        
        # Cargar datos
        print("Cargando datos...")
        train_loader, val_loader, test_loader, class_names = load_data(
            args.data_path, processor, args.batch_size, args.train_split, args.val_split
        )
        
        num_classes = len(class_names)
        print(f"Clases encontradas: {class_names}")
        print(f"Número de clases: {num_classes}")
        
        # Información del dataset
        total_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        print(f"Total de muestras: {total_samples}")
        print(f"Entrenamiento: {len(train_loader.dataset)}")
        print(f"Validación: {len(val_loader.dataset)}")  
        print(f"Prueba: {len(test_loader.dataset)}")
        
        # Log de parámetros
        mlflow.log_param("device", str(device))
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("class_names", class_names)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("train_split", args.train_split)
        mlflow.log_param("val_split", args.val_split)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("total_samples", total_samples)
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))
        mlflow.log_param("test_samples", len(test_loader.dataset))
        
        # Crear modelo
        print("Creando modelo ViT...")
        model = create_vit_model(args.model_name, num_classes, class_names)
        model = model.to(device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        
        # Optimizador
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        
        # Entrenar modelo
        print("Iniciando entrenamiento...")
        train_losses, val_accs, best_val_acc = train_model(
            model, train_loader, val_loader, optimizer, device, args.epochs
        )
        
        # Log métricas finales
        mlflow.log_metric("final_best_val_accuracy", best_val_acc)
        mlflow.log_metric("final_train_loss", train_losses[-1])
        
        # Evaluación en test set
        print("Evaluando en conjunto de prueba...")
        test_accuracy = evaluate_model(model, test_loader, device)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"Exactitud en prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Evaluación detallada
        print("Evaluación detallada...")
        predictions, targets, _, report = detailed_evaluation(
            model, test_loader, class_names, device
        )
        
        # Calcular exactitud por clase
        per_class_acc = calculate_per_class_accuracy(predictions, targets, class_names)
        
        # Log métricas por clase
        for class_name, acc in per_class_acc.items():
            mlflow.log_metric(f"accuracy_{class_name}", acc)
        
        
        # Guardar modelo y processor
        #model_dir = "vit_finetuned"
        #os.makedirs(model_dir, exist_ok=True)
        #model.save_pretrained(model_dir)
        #processor.save_pretrained(model_dir)
        
        # Log del modelo en MLFlow
        #mlflow.transformers.log_model(
        #    transformers_model={
        #        "model": model,
        #        "tokenizer": processor  # ViT usa processor en lugar de tokenizer
        #    },
        #    artifact_path="model",
        #    registered_model_name=f"vit_classifier_{args.experiment_name}"
        #)
        
        print(f"Experimento completado. Mejor exactitud: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()

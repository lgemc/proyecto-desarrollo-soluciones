# Animal Classification

Aplicación web para clasificación de imágenes de animales usando ResNet50 con interfaz React y API FastAPI.

## 🚀 Ejecución Rápida con Docker

### Prerrequisitos

1. **Git LFS** (para descargar assets del frontend):
   ```bash
   # macOS
   brew install git-lfs
   
   # Ubuntu/Debian
   sudo apt install git-lfs
   
   # Configurar Git LFS
   git lfs install
   git lfs pull
   ```

2. **Modelo entrenado** (generar el archivo `models/animal-classifier-resnet.pth`):
   ```bash
   # Instalar dependencias de desarrollo
   uv sync --group dev
   
   # Entrenar modelo ResNet50 (ejemplo con 10 épocas)
   uv run python animal_classification/train/train_resnet_mlflow.py \
     --data_path data \
     --epochs 10 \
     --batch_size 16 \
   ```

### Construcción y Ejecución

```bash
# Construir la imagen Docker
docker build -f infra/Dockerfile --tag proyecto-desarrollo-soluciones .

# Ejecutar la aplicación
docker run -p 8000:8000 proyecto-desarrollo-soluciones
```

La aplicación estará disponible en: **http://localhost:8000**

### Endpoints de la API

- **Health Check**: `GET /api/v1/health`
- **Clasificación**: `POST /api/v1/classify` (multipart/form-data con campo `image`)

Ejemplo de uso:
```bash
curl -X POST -F "image=@data/Zebra/Zebra_298.jpg" http://localhost:8000/api/v1/classify
```

## 🐳 Arquitectura del Docker

El `Dockerfile` utiliza un **build multi-stage** optimizado para producción:

### Stage 1: Frontend Builder (`frontend-builder`)
```dockerfile
FROM node:20-alpine AS frontend-builder
```
- **Propósito**: Construir la interfaz web React/TypeScript
- **Proceso**: 
  - Instala dependencias con `npm ci`
  - Ejecuta `npm run build` con Vite
  - Genera archivos estáticos optimizados en `/frontend/dist`
- **Output**: Archivos HTML, CSS, JS y assets optimizados

### Stage 2: Model Converter (`model-converter`)
```dockerfile
FROM python:3.12-slim AS model-converter
```
- **Propósito**: Convertir modelo PyTorch a formato ONNX para inferencia optimizada
- **Proceso**:
  - Instala dependencias completas de ML (PyTorch, ONNX, etc.)
  - Copia el modelo `.pth` entrenado
  - Ejecuta `tools/convert_to_onnx.py` para generar `.onnx` y metadatos
- **Output**: 
  - `animal-classifier-resnet.onnx` (modelo optimizado)
  - `animal-classifier-resnet.json` (metadatos: clases, forma de entrada)

### Stage 3: Production (`production`)
```dockerfile
FROM python:3.12-slim AS production
```
- **Propósito**: Imagen final ligera solo con dependencias de runtime
- **Proceso**:
  - Copia código de la aplicación
  - Copia modelo ONNX desde `model-converter`
  - Copia frontend construido desde `frontend-builder`
  - Instala solo dependencias mínimas de producción
- **Configuración**:
  - Puerto: `8000`
  - Comando: `uvicorn animal_classification.app.main:app`
  - Variables de entorno optimizadas para Python


## 🏗️ Estructura del Proyecto

```
proyecto-desarrollo-soluciones/
├── animal_classification/          # Paquete principal de Python
│   ├── app/                       # API FastAPI
│   │   └── main.py               # Servidor web y endpoints
│   ├── inference/                # Módulos de inferencia
│   │   ├── onnx_classifier.py    # Clasificador ONNX optimizado
│   │   └── resnet_classifier.py  # Clasificador PyTorch
│   ├── models/                   # Definiciones de arquitecturas
│   ├── preprocessing/            # Preprocesamiento de imágenes
│   ├── train/                    # Scripts de entrenamiento
│   │   ├── train_resnet_mlflow.py # Entrenamiento ResNet50 
│   │   └── train_vit_mlflow.py   # Entrenamiento Vision Transformer
│   └── utils/                    # Utilidades compartidas
├── ui/                           # Frontend React/TypeScript
│   ├── src/
│   │   ├── components/          # Componentes React
│   │   ├── pages/              # Páginas principales
│   │   ├── shared/             # API client y utilidades
│   │   └── assets/             # Imágenes y recursos estáticos
│   ├── package.json
│   └── vite.config.ts          # Configuración del bundler
├── infra/                       # Infraestructura y deployment
│   ├── Dockerfile              # Definición de contenedor
│   ├── requirements.txt        # Dependencias de producción
│   └── aws/                    # Configuración Terraform (AWS)
├── models/                     # Modelos entrenados y metadatos
├── data/                       # Dataset (gestionado con DVC)
├── notebooks/                  # Jupyter notebooks para experimentación
└── tools/                      # Scripts de utilidades
    └── convert_to_onnx.py     # Conversión PyTorch → ONNX
```

## 🔧 Desarrollo Local

### Configuración del Entorno

```bash
# Crear entorno virtual con uv
uv venv --python 3.11

# Instalar dependencias de desarrollo
uv sync --group dev

# Instalar hooks de pre-commit
pre-commit install
```

### Comandos de Desarrollo

```bash
# Ejecutar servidor FastAPI (modo desarrollo)
make app-run
# o directamente:
uv run uvicorn animal_classification.app.main:app --reload

# Ejecutar tests
make test

# Linting y formateo
make lint
make format

# Ejecutar todos los checks
make check
```

### Entrenamiento de Modelos

```bash
# ResNet50 (recomendado para producción)
uv run python animal_classification/train/train_resnet_mlflow.py \
  --data_path data \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0005

# Vision Transformer (experimental)
uv run python animal_classification/train/train_vit_mlflow.py \
  --data_path data \
  --epochs 5 \
  --batch_size 16
```

### Frontend (Desarrollo)

```bash
cd ui
npm install
npm run dev  # Servidor de desarrollo en puerto 3000
npm run build  # Build para producción
```

## 📊 Características del Modelo

- **Arquitectura**: ResNet50 pre-entrenado (ImageNet) + fine-tuning
- **Clases**: Buffalo, Elephant, Rhino, Zebra
- **Precisión**: >97% en conjunto de prueba
- **Formato de producción**: ONNX (optimizado para inferencia)
- **Preprocesamiento**: Resize(256) → CenterCrop(224) → Normalización ImageNet

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI**: API REST moderna y rápida
- **PyTorch**: Framework de deep learning
- **ONNX Runtime**: Inferencia optimizada
- **MLflow**: Tracking de experimentos
- **Uvicorn**: Servidor ASGI de alto rendimiento

### Frontend
- **React 18**: Biblioteca de interfaz de usuario
- **TypeScript**: JavaScript tipado
- **Vite**: Build tool moderno y rápido
- **Tailwind CSS**: Framework de estilos utilitarios

### DevOps
- **Docker**: Containerización multi-stage
- **uv**: Gestor de dependencias Python rápido
- **Pre-commit**: Hooks de calidad de código
- **Git LFS**: Gestión de archivos grandes

## 📝 Notas Importantes

- Los datos están gestionados con **DVC** y no se versionan en Git
- Los assets del frontend requieren **Git LFS** para descargarse
- El modelo se entrena localmente y se convierte a ONNX durante el build
- La aplicación sirve tanto la API como el frontend desde el mismo puerto (8000)
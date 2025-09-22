# Manual de uso de la api

La aplicación api tiene un endpoint que permite la clasificación de imagenes
enviando una petición post con la imagen a clasificar.

## Endpoint

`POST /api/v1/classify`

### Parámetros

- `image`: Imagen a clasificar (formato multipart/form-data).

### Respuesta

- `classification_label`: Etiqueta de clasificación (nombre del animal).
- `class`: Clase de clasificación (número entero).
- `confidence`: Confianza de la clasificación (valor entre 0 y 1).

Esta respuesta se entrega en formato JSON.

### Ejemplo de uso con `curl`

```bash
curl --location 'http://0.0.0.0:8000/api/v1/classify' \
--header 'accept: application/json' \
--form 'image=@"/home/lmanrique/Do/proyecto-desarrollo-soluciones/data/Rhino/Rhino_46.jpg"'
```

### Respuesta de ejemplo

```json
{
    "classification_label": "Rhino",
    "class": 2,
    "confidence": 0.9998983144760132
}
```

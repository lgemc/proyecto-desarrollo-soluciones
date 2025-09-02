# Animal classifier: API

As part of the whole animal classification project, this module provides a simple API to classify images of animals using a pre-trained model.

# Supported Animals:

- Buffalo (0)
- Elephant (1)
- Rhino (2)
- Zebra (3)

# Usage

We have two main endpoints:

- `GET /api/v1/health`: Health check endpoint to verify the service is running.
- `POST /api/v1/predict`: Endpoint to classify an image of an animal.

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/api/v1/classify" \                                                                                                   (base) 
      -H "accept: application/json" \
      -H "Content-Type: multipart/form-data" \
      -F "image=@/.../data/Rhino/Rhino_1000.jpg"
```

Response:

```json 
{"classification_label":"Rhino","class":2,"confidence":0.9979546070098877}
``` 

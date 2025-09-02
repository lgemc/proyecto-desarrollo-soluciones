import { useState } from 'react';
import { classifyImage, ClassificationResponse } from '../shared/api/classify';

export default function ImageUpload() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState<ClassificationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setClassificationResult(null);
      setError(null);
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) return;
    
    setIsClassifying(true);
    setError(null);
    
    try {
      const result = await classifyImage(selectedImage);
      setClassificationResult(result);
    } catch (err) {
      setError('Error clasificando la imagen. Por favor, intenta de nuevo.');
      console.error('Classification error:', err);
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="bg-white bg-opacity-20 rounded-lg p-8 text-center">
      {previewUrl ? (
        <div className="mb-4">
          <img 
            src={previewUrl} 
            alt="Imagen seleccionada" 
            className="max-w-full max-h-48 mx-auto rounded"
          />
        </div>
      ) : (
        <div className="mb-4">
          <div className="bg-yellow-400 w-8 h-8 rounded-full mb-2 mx-auto"></div>
          <div className="bg-green-500 w-12 h-8 rounded mb-2 mx-auto"></div>
          <div className="bg-blue-500 w-8 h-8 rounded-full flex items-center justify-center mx-auto">
            <span className="text-white text-xs">↑</span>
          </div>
        </div>
      )}
      
      <p className="text-xs mb-4">Selecciona una imagen de un<br/>animal para clasificarlo.</p>
      
      <input
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        onChange={handleFileSelect}
        className="hidden"
        id="image-upload"
      />
      
      <div className="space-y-2">
        <label
          htmlFor="image-upload"
          className="bg-blue-500 text-white px-4 py-2 rounded text-sm cursor-pointer hover:bg-blue-600 inline-block"
        >
          Seleccionar imagen
        </label>
        
        {selectedImage && (
          <button 
            onClick={handleClassify}
            disabled={isClassifying}
            className="bg-green-200 text-green-800 px-4 py-2 rounded text-sm block mx-auto hover:bg-green-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isClassifying ? 'Clasificando...' : 'Clasificar animal'}
          </button>
        )}
        
        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
        
        {classificationResult && (
          <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
            <h3 className="font-semibold mb-2">Resultado de clasificación:</h3>
            <p><strong>Animal:</strong> {classificationResult.classification_label}</p>
            <p><strong>Clase:</strong> {classificationResult.class}</p>
            <p><strong>Confianza:</strong> {(classificationResult.confidence * 100).toFixed(1)}%</p>
          </div>
        )}
      </div>
    </div>
  );
}
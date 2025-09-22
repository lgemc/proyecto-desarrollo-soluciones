import { useState } from 'react';
import { classifyImage, ClassificationResponse } from '../shared/api/classify';

interface ImageUploadProps {
  onImageSelect?: (file: File, previewUrl: string) => void;
  onClassificationResult?: (result: ClassificationResponse) => void;
}

export default function ImageUpload({ onImageSelect, onClassificationResult }: ImageUploadProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setError(null);
      onImageSelect?.(file, url);
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) return;
    
    setIsClassifying(true);
    setError(null);
    
    try {
      const result = await classifyImage(selectedImage);
      onClassificationResult?.(result);
      setSelectedImage(null);
    } catch (err) {
      setError('Error clasificando la imagen. Por favor, intenta de nuevo.');
      console.error('Classification error:', err);
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="bg-white bg-opacity-20 rounded-lg p-4 sm:p-8 text-center">
      <p className="text-xs mb-4 px-2">Selecciona una imagen de un<br/>animal para clasificarlo.</p>
      
      <input
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        onChange={handleFileSelect}
        className="hidden"
        id="image-upload"
      />
      
      <div className="space-y-2">
        {!selectedImage ? (
          <label
            htmlFor="image-upload"
            className="bg-green-200 text-green-800 px-3 py-2 sm:px-4 rounded text-sm cursor-pointer hover:bg-green-300 inline-block"
          >
            Seleccionar imagen
          </label>
        ) : (
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
      </div>
    </div>
  );
}
import { useState } from 'react';
import LeftPanel from '../components/LeftPanel';
import Header from '../components/Header';
import ResultDisplay from '../components/ResultDisplay';
import { ClassificationResponse } from '../shared/api/classify';
import { translateAnimal } from '../shared/translations';

export default function FaunaScan() {
  const [selectedImageUrl, setSelectedImageUrl] = useState<string | null>(null);
  const [classificationResult, setClassificationResult] = useState<ClassificationResponse | null>(null);

  const handleImageSelect = (file: File, previewUrl: string) => {
    setSelectedImageUrl(previewUrl);
    setClassificationResult(null);
  };

  const handleClassificationResult = (result: ClassificationResponse) => {
    setClassificationResult(result);
  };

  const resultForDisplay = classificationResult ? {
    image: selectedImageUrl || '',
    name: translateAnimal(classificationResult.classification_label),
    alt: translateAnimal(classificationResult.classification_label)
  } : undefined;

  return (
    <div className="bg-gray-100 min-h-screen p-2 sm:p-4">
      <Header />
      <div className="max-w-6xl mx-auto bg-white rounded-lg overflow-hidden shadow-lg flex flex-col md:flex-row">
        <LeftPanel 
          onImageSelect={handleImageSelect}
          onClassificationResult={handleClassificationResult}
        />
        <ResultDisplay 
          selectedImage={selectedImageUrl}
          result={resultForDisplay}
        />
      </div>
    </div>
  );
}
import ImageUpload from './ImageUpload';
import ImagePreview from "../assets/image-preview.png";
import { ClassificationResponse } from '../shared/api/classify';

interface LeftPanelProps {
  onImageSelect?: (file: File, previewUrl: string) => void;
  onClassificationResult?: (result: ClassificationResponse) => void;
}

export default function LeftPanel({ onImageSelect, onClassificationResult }: LeftPanelProps) {
  return (
    <div className="p-4 sm:p-6 flex flex-col items-center justify-center w-full md:w-1/2">
      <div className="bg-white p-2 rounded-lg mb-4">
        <div className="w-48 h-48 sm:w-60 sm:h-60 rounded flex items-center justify-center">
          <img src={ImagePreview} alt="Logo" className="max-w-full max-h-full" />
        </div>
      </div>
      <ImageUpload 
        onImageSelect={onImageSelect}
        onClassificationResult={onClassificationResult}
      />
    </div>
  );
}
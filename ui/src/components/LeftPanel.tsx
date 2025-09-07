import ImageUpload from './ImageUpload';
import ImagePreview from "../assets/image-preview.png";
import { ClassificationResponse } from '../shared/api/classify';

interface LeftPanelProps {
  onImageSelect?: (file: File, previewUrl: string) => void;
  onClassificationResult?: (result: ClassificationResponse) => void;
}

export default function LeftPanel({ onImageSelect, onClassificationResult }: LeftPanelProps) {
  return (
    <div className="p-6 flex flex-col items-center justify-center w-1/2">
      <div className="bg-white p-2 rounded-lg mb-4">
        <div className="w-60 h-60 rounded flex items-center justify-center">
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
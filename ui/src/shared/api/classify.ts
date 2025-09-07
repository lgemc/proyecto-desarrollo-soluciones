import { postFormData } from './base';

export interface ClassificationResponse {
  classification_label: string;
  class: number;
  confidence: number;
}

export async function classifyImage(imageFile: File): Promise<ClassificationResponse> {
  const formData = new FormData();
  formData.append('image', imageFile);

  return postFormData<ClassificationResponse>(
    '/api/v1/classify',
    formData,
  );
}
interface ResultDisplayProps {
  result?: {
    image: string;
    name: string;
    alt: string;
  };
  selectedImage?: string;
}

const defaultResult = {
  image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgdmlld0JveD0iMCAwIDE1MCAxNTAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxNTAiIGhlaWdodD0iMTUwIiBmaWxsPSIjRjNGNEY2Ii8+CjxyZWN0IHg9IjIwIiB5PSI0MCIgd2lkdGg9IjExMCIgaGVpZ2h0PSI3MCIgZmlsbD0iIzlDQTNBRiIvPgo8L3N2Zz4K",
  name: "Cebra",
  alt: "Zebra"
};

export default function ResultDisplay({ result, selectedImage }: ResultDisplayProps) {
  const displayImage = selectedImage || defaultResult.image;
  
  return (
    <div className="w-1/2 p-6 flex flex-col items-center justify-center border-l">
      <h2 className="text-center text-gray-600 font-semibold mb-4">Resultado</h2>
      <div className="text-center">
        <img 
          src={displayImage}
          alt={result?.alt || defaultResult.alt}
          className="w-32 h-32 mx-auto mb-4 rounded object-cover" 
        />
        {result ? (
          <h3 className="text-2xl font-semibold text-gray-800">{result.name}</h3>
        ) : (
          <h3 className="text-2xl font-semibold text-gray-400">-</h3>
        )}
      </div>
    </div>
  );
}
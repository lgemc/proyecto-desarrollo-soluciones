export default function Header() {
  return (
    <div className="bg-green-600 text-white p-6 flex flex-col items-center justify-center w-1/2">
      <div className="bg-white p-2 rounded-lg mb-4">
        <div className="w-8 h-8 bg-green-600 rounded flex items-center justify-center">
          <span className="text-white text-sm">🦁</span>
        </div>
      </div>
      <h1 className="text-2xl font-bold mb-2">FAUNA</h1>
      <p className="text-sm mb-1">SCAN</p>
      <p className="text-xs text-center mb-6">Herramienta de<br/>Clasificación de animales</p>
    </div>
  );
}
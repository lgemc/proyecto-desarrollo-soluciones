import LeftPanel from '../components/LeftPanel';
import ResultDisplay from '../components/ResultDisplay';

export default function FaunaScan() {
  return (
    <div className="bg-gray-100 min-h-screen p-4">
      <div className="max-w-4xl mx-auto bg-white rounded-lg overflow-hidden shadow-lg flex">
        <LeftPanel />
        <ResultDisplay />
      </div>
    </div>
  );
}
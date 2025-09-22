import Snitch from "../assets/snitch.png";

export default function Header() {
    return (
        <div className="bg-green-600 text-white flex flex-col sm:flex-row justify-between items-center px-4 sm:px-8 py-2"> 
            <div className="flex items-center">
                <div className="flex items-center justify-center p-2">
                    <img src={Snitch} alt="Logo" className="h-12 w-12"/>
                </div>
                <div className="flex items-center justify-center flex-col p-2 sm:p-4">
                    <h1 className="text-xl sm:text-2xl font-bold mb-1 sm:mb-2">FAUNA</h1>
                    <h1 className="text-sm sm:text-base mb-1">SCAN</h1>
                </div>
            </div>
            <div className="flex items-center justify-center flex-col p-2 sm:p-4">
                <p className="text-xs text-center">Herramienta de<br/>Clasificación de animales</p>
            </div>
        </div>
    );
}
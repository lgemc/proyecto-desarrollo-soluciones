import Snitch from "../assets/snitch.png";

export default function Header() {
    return (
        <div className="bg-green-600 text-white flex justify-between pl-8 pr-8"> 
            <div className="flex items-center">
                <div className="flex items-center justify-center p-2">
                    <img src={Snitch} alt="Logo" className="h-12 w-12"/>
                </div>
                <div className="flex items-center justify-center flex-col p-4">
                    <h1 className="text-2xl font-bold mb-2">FAUNA</h1>
                    <h1 className="text-1xl mb-1">SCAN</h1>
                </div>
            </div>
            <div className="flex items-center justify-center flex-col p-4">
                <p className="text-xs">Herramienta de<br/>Clasificación de animales</p>
            </div>
        </div>
    );
}
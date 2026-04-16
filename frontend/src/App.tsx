import { useState } from 'react';
import Navbar from './components/Navbar';
import CameraBox from './components/CameraBox';
import OutputPanel from './components/OutputPanel';
import './index.css';

function App() {
  const [current, setCurrent] = useState("");
  const [history, setHistory] = useState<string[]>([]);

  const handlePrediction = (text: string) => {
    setCurrent(text);
    setHistory((prev) => {
      // Avoid repeating the same word infinitely
      if (prev[prev.length - 1] === text) return prev;
      return [...prev, text];
    });
  };

  return (
  <div className="min-h-screen bg-[#f3f4f6] font-sans" >
    <Navbar />
    
    {/* This main tag creates the 3-column layout */}
    <main className="max-w-[1400px] mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
      
      {/* Left Column (Video) - Spans 2 out of 3 columns */}
      <div className="lg:col-span-2 flex flex-col gap-6">
        <CameraBox onPrediction={handlePrediction} />
        
        <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm flex items-center gap-3">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></div>
          <p className="text-gray-600 text-sm">
            <b>Pro Tip:</b> Ensure your hand is well-lit for the LSTM model to extract keypoints.
          </p>
        </div>
      </div>

      {/* Right Column (Output) - Spans 1 out of 3 columns */}
      <div className="lg:col-span-1">
        <OutputPanel 
          current={current} 
          history={history} 
          clearHistory={() => setHistory([])} 
        />
      </div>
    </main>
  </div>
);
}

export default App;
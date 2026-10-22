import { useState } from 'react';
import Navbar from './components/Navbar';
import CameraBox from './components/CameraBox';
import OutputPanel from './components/OutputPanel';
import './index.css';

function App() {
  const [currentLetter, setCurrentLetter] = useState("");
  const [currentProb, setCurrentProb] = useState<number>(0);
  const [currentWord, setCurrentWord] = useState("");
  const [sentence, setSentence] = useState<string>("");
  const lastAppendRef = useState<{ letter: string; at: number }>({ letter: "", at: 0 })[0];

  const handlePrediction = (text: string, probability?: number) => {
    const cleaned = text.trim();
    setCurrentProb(typeof probability === "number" ? probability : 0);

    if (!cleaned) {
      setCurrentLetter("");
      return;
    }

    // Alphabet model outputs A-Z only. We build words client-side.
    const letter = cleaned.toUpperCase();
    setCurrentLetter(letter);

    const now = Date.now();
    const cooldownMs = 700;
    if (lastAppendRef.letter === letter && now - lastAppendRef.at < cooldownMs) return;
    lastAppendRef.letter = letter;
    lastAppendRef.at = now;

    setCurrentWord((prev) => prev + letter);
  };

  return (
  <div className="min-h-screen font-sans bg-gradient-to-b from-slate-50 via-blue-50/40 to-slate-50">
    <Navbar />
    
    {/* This main tag creates the 3-column layout */}
    <main className="max-w-[1400px] mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
      
      {/* Left Column (Video) - Spans 2 out of 3 columns */}
      <div className="lg:col-span-2 flex flex-col gap-6">
        <CameraBox onPrediction={handlePrediction} />
        
        <div className="bg-white/80 backdrop-blur p-4 rounded-2xl border border-white shadow-sm flex items-start gap-3">
          <div className="mt-1 w-2.5 h-2.5 bg-blue-600 rounded-full"></div>
          <p className="text-gray-700 text-sm leading-relaxed">
            <b className="text-gray-900">Pro Tip:</b> Keep your hand centered and steady in good lighting. For words, hold the motion for a full second.
          </p>
        </div>
      </div>

      {/* Right Column (Output) - Spans 1 out of 3 columns */}
      <div className="lg:col-span-1">
        <OutputPanel 
          current={currentLetter} 
          currentProb={currentProb}
          currentWord={currentWord}
          sentence={sentence}
          onCommitWord={() => {
            if (!currentWord) return;
            setSentence((prev) => (prev ? `${prev} ${currentWord}` : currentWord));
            setCurrentWord("");
          }}
          onBackspace={() => {
            setCurrentWord((prev) => prev.slice(0, -1));
          }}
          onPunct={(p) => {
            setSentence((prev) => (prev ? `${prev}${p}` : p));
          }}
          clearAll={() => {
            setCurrentWord("");
            setSentence("");
          }}
        />
      </div>
    </main>
  </div>
);
}

export default App;
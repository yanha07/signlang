interface OutputProps {
  current: string;
  currentProb: number;
  currentWord: string;
  sentence: string;
  onCommitWord: () => void;
  onBackspace: () => void;
  onPunct: (p: string) => void;
  clearAll: () => void;
}

export default function OutputPanel({ current, currentProb, currentWord, sentence, onCommitWord, onBackspace, onPunct, clearAll }: OutputProps) {
  const prob = Number.isFinite(currentProb) ? Math.max(0, Math.min(100, currentProb)) : 0;
  const isLowConfidence = current.length > 0 && prob > 0 && prob < 60;

  return (
    <div className="flex flex-col gap-6 h-full">
    <div className="bg-white/70 backdrop-blur border border-white rounded-2xl p-4 shadow-sm">
      <p className="text-sm text-gray-900 font-semibold">How to use</p>
      <p className="text-sm text-gray-600 mt-1 leading-relaxed">
        Press <b>Start</b>, show one sign clearly for 1–2 seconds, then pause before the next sign.
      </p>
    </div>
    {/* Current Prediction Card */}
    <div className="bg-white p-8 rounded-[2rem] shadow-xl border border-gray-100">
      <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-4">Live Translation</h3>
      <div className="text-6xl sm:text-7xl font-black text-blue-600 text-center py-6">
        {current || "Waiting..."}
      </div>

      <div className="mt-2">
        <div className="flex items-center justify-between text-xs font-semibold text-gray-600">
          <span>Confidence</span>
          <span>{current ? `${prob.toFixed(1)}%` : "--"}</span>
        </div>
        <div className="mt-2 h-2 w-full rounded-full bg-gray-100 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 transition-all"
            style={{ width: current ? `${prob}%` : "0%" }}
          />
        </div>
        {isLowConfidence && (
          <p className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-100 rounded-xl px-3 py-2">
            Low confidence — hold the sign steady and keep your hand centered in the frame.
          </p>
        )}
      </div>
    </div>

    {/* History Card */}
    <div className="bg-white p-8 rounded-[2rem] shadow-lg border border-gray-100 flex-grow">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Sentence Builder</h3>
        <button type="button" onClick={clearAll} className="text-xs font-semibold text-blue-600 hover:text-blue-700">
          Clear
        </button>
      </div>

      <div className="space-y-4">
        <div className="rounded-2xl border border-gray-100 bg-gray-50 p-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">Current word</p>
          <p className="mt-2 text-3xl font-extrabold text-gray-900 break-words">{currentWord || "—"}</p>
          <div className="mt-4 flex flex-wrap gap-2">
            <button type="button" onClick={onCommitWord} className="px-4 py-2 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-700">
              Add word
            </button>
            <button type="button" onClick={onBackspace} className="px-4 py-2 rounded-xl bg-white border border-gray-200 font-semibold text-gray-800 hover:bg-gray-50">
              Backspace
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-gray-100 bg-white p-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">Sentence</p>
          <p className="mt-2 text-2xl font-medium text-gray-700 leading-relaxed break-words">
            {sentence || "Your words will appear here."}
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {[".", ",", "?", "!", ":"].map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => onPunct(p)}
                className="w-10 h-10 rounded-xl bg-gray-50 border border-gray-200 font-black text-gray-700 hover:bg-gray-100"
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}
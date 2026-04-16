interface OutputProps {
  current: string;
  history: string[];
  clearHistory: () => void;
}

export default function OutputPanel({ current, history, clearHistory }: OutputProps) {
  return (
    <div className="flex flex-col gap-6 h-full">
    {/* Current Prediction Card */}
    <div className="bg-white p-8 rounded-[2rem] shadow-xl border border-blue-50">
      <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-4">Live Detection</h3>
      <div className="text-7xl font-black text-blue-600 text-center py-4">
        {current || "..."}
      </div>
    </div>

    {/* History Card */}
    <div className="bg-white p-8 rounded-[2rem] shadow-lg border border-gray-100 flex-grow">
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Translation History</h3>
      <p className="text-2xl font-medium text-gray-700 leading-relaxed">
        {history.length > 0 ? history.join(" ") : "Waiting for gesture..."}
      </p>
    </div>
  </div>
  );
}
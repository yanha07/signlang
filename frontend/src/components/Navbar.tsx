export default function Navbar() {
  return (
    <nav className="sticky top-0 z-20 backdrop-blur bg-white/80 border-b border-gray-200">
      <div className="px-6 py-4 max-w-[1400px] mx-auto flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-sm flex items-center justify-center">
            <span className="text-white font-black text-lg">SB</span>
          </div>
          <div className="leading-tight">
            <h1 className="text-lg sm:text-xl font-black text-gray-900">Sign Bridge</h1>
            <p className="text-xs text-gray-500">Sign language to English</p>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5">
          <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse"></span>
          <span className="text-xs sm:text-sm font-semibold text-emerald-800">Ready</span>
        </div>
      </div>
    </nav>
  );
}
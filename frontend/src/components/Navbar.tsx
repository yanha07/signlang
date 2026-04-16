export default function Navbar() {
  return (
    <nav className="p-4 bg-white border-b border-gray-200 flex justify-between items-center px-8">
      <h1 className="text-xl font-bold text-blue-600">Sign Bridge</h1>
      <div className="flex items-center gap-2">
        <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
        <span className="text-sm font-medium text-gray-600">System Ready</span>
      </div>
    </nav>
  );
}
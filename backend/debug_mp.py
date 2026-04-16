import importlib
try:
    import mediapipe.python.solutions
except Exception as e:
    import traceback
    traceback.print_exc()

import mediapipe as mp
print(dir(mp))
try:
    print("Trying mp.solutions:", mp.solutions)
except Exception as e:
    print(e)
try:
    from mediapipe.python.solutions import holistic
    print("Imported holistic directly:", holistic)
except Exception as e:
    print("Direct import failed:", e)

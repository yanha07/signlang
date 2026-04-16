import { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

declare global {
  interface Window {
    Holistic: any;
    Camera: any;
    drawConnectors: any;
    drawLandmarks: any;
    FACEMESH_TESSELATION: any;
    POSE_CONNECTIONS: any;
    HAND_CONNECTIONS: any;
  }
}

interface CameraProps {
  onPrediction: (text: string) => void;
}

export default function CameraBox({ onPrediction }: CameraProps) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  
  const lastPredictionRef = useRef({ text: "", prob: 0, timeout: null as any });
  const sequenceRef = useRef<number[][]>([]);
  const isRecordingRef = useRef(false);
  const frameCountRef = useRef(0);

  useEffect(() => {
    isRecordingRef.current = isRecording;
    if (!isRecording) {
      sequenceRef.current = [];
      frameCountRef.current = 0;
    }
  }, [isRecording]);

  useEffect(() => {
    if (!window.Holistic || !window.Camera || !window.drawConnectors) {
      console.error("MediaPipe not loaded fully. Please check internet connection for CDN.");
      return;
    }

    const holistic = new window.Holistic({
      locateFile: (file: string) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      }
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      selfieMode: false
    });

    holistic.onResults((results: any) => {
      // 1. Draw the landmarks so the user can literally SEE what the model sees!
      drawVisibleSkeleton(results);

      if (!isRecordingRef.current) return;
      
      const keypoints = extractKeypoints(results);
      
      frameCountRef.current++;
      
      // Increased time: We now only predict once every 3-4 seconds (every 100 frames).
      // This gives you plenty of time to position your hand before the picture is snapped!
      if (frameCountRef.current % 100 === 0) {
        // HACK: The Kaggle dataset used was static images copied 30 times.
        // We will duplicate this exact single frame 30 times to perfectly mimic the training data!
        const frozenSequence = new Array(30).fill(keypoints);
        
        fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sequence: frozenSequence }),
        })
        .then(res => res.json())
        .then(data => {
            if (data.prediction && data.prediction.trim() !== '') {
                onPrediction(data.prediction);
                
                // Store the prediction to draw the OpenCV-style green box
                lastPredictionRef.current.text = data.prediction;
                lastPredictionRef.current.prob = data.probability;
                
                if (lastPredictionRef.current.timeout) {
                    clearTimeout(lastPredictionRef.current.timeout);
                }
                lastPredictionRef.current.timeout = setTimeout(() => {
                    lastPredictionRef.current.text = "";
                }, 4000); // Box disappears after 4 seconds
            }
        })
        .catch(err => {
            console.error("Prediction fetch error:", err);
        });
      }
    });

    let camera: any = null;

    if (webcamRef.current && webcamRef.current.video) {
        camera = new window.Camera(webcamRef.current.video, {
            onFrame: async () => {
                if (webcamRef.current?.video) {
                    await holistic.send({image: webcamRef.current.video});
                }
            },
            width: 640,
            height: 480
        });
        camera.start();
    }

    return () => {
      if (camera) {
        camera.stop();
      }
    };
  }, [onPrediction]);

  // Visual skeleton drawing logic
  const drawVisibleSkeleton = (results: any) => {
    const canvasCtx = canvasRef.current?.getContext('2d');
    const canvasEl = canvasRef.current;
    if (!canvasCtx || !canvasEl) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    // Draw Face
    if (results.faceLandmarks) {
      window.drawConnectors(canvasCtx, results.faceLandmarks, window.FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
    }
    // Draw Pose
    if (results.poseLandmarks) {
      window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {color: 'white', lineWidth: 2});
      window.drawLandmarks(canvasCtx, results.poseLandmarks, {color: 'red', lineWidth: 1});
    }
    // Draw Left Hand
    if (results.leftHandLandmarks) {
      window.drawConnectors(canvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
      window.drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#FF0000', lineWidth: 1});
    }
    // Draw Right Hand
    if (results.rightHandLandmarks) {
      window.drawConnectors(canvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {color: '#0000FF', lineWidth: 2});
      window.drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 1});
    }
    
    // Draw OpenCV-Style Bounding Box with Probability (just like Python cv2.putText)
    if (lastPredictionRef.current.text !== "") {
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        let hasHands = false;
        
        const updateBounds = (lm: any) => {
            if (lm.x < minX) minX = lm.x;
            if (lm.y < minY) minY = lm.y;
            if (lm.x > maxX) maxX = lm.x;
            if (lm.y > maxY) maxY = lm.y;
            hasHands = true;
        };
        
        if (results.leftHandLandmarks) results.leftHandLandmarks.forEach(updateBounds);
        if (results.rightHandLandmarks) results.rightHandLandmarks.forEach(updateBounds);
        
        if (hasHands) {
            const cw = canvasEl.width;
            const ch = canvasEl.height;
            
            // Convert normalized coordinates back to canvas pixels with slight padding
            const px_minX = Math.max(0, minX - 0.05) * cw;
            const px_minY = Math.max(0, minY - 0.05) * ch;
            const px_maxX = Math.min(1, maxX + 0.05) * cw;
            const px_maxY = Math.min(1, maxY + 0.05) * ch;
            
            // 1. Draw glowing green box around hands
            canvasCtx.beginPath();
            canvasCtx.rect(px_minX, px_minY, px_maxX - px_minX, px_maxY - px_minY);
            canvasCtx.lineWidth = 4;
            canvasCtx.strokeStyle = '#00FF00';
            canvasCtx.stroke();
            
            // 2. Format the Text ('Hello: 99.0%')
            const textStr = `${lastPredictionRef.current.text}: ${lastPredictionRef.current.prob.toFixed(1)}%`;
            canvasCtx.font = 'bold 24px Arial';
            const textWidth = canvasCtx.measureText(textStr).width;
            const textHeight = 30; 
            
            // 3. Draw Green Text Background Rectangle directly above the bounding box
            const bg_Y = px_minY - textHeight;
            canvasCtx.fillStyle = '#00FF00';
            canvasCtx.fillRect(px_minX, bg_Y, textWidth + 10, textHeight);
            
            // 4. Draw Black Text inside the rectangle
            canvasCtx.fillStyle = '#000000';
            canvasCtx.fillText(textStr, px_minX + 5, px_minY - 6);
        }
    }
    
    canvasCtx.restore();
  };

  const extractKeypoints = (results: any) => {
    let pose = new Array(132).fill(0);
    if (results.poseLandmarks) {
      pose = results.poseLandmarks.map((res: any) => [res.x || 0, res.y || 0, res.z || 0, res.visibility || 0]).flat();
    }
    
    let face = new Array(1404).fill(0);
    if (results.faceLandmarks) {
      face = results.faceLandmarks.slice(0, 468).map((res: any) => [res.x || 0, res.y || 0, res.z || 0]).flat();
    }
    
    let lh = new Array(63).fill(0);
    if (results.leftHandLandmarks) {
      lh = results.leftHandLandmarks.map((res: any) => [res.x || 0, res.y || 0, res.z || 0]).flat();
    }
    
    let rh = new Array(63).fill(0);
    if (results.rightHandLandmarks) {
      rh = results.rightHandLandmarks.map((res: any) => [res.x || 0, res.y || 0, res.z || 0]).flat();
    }

    return [...pose, ...face, ...lh, ...rh];
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="relative w-full h-[500px] bg-black rounded-2xl overflow-hidden border-4 border-white shadow-xl flex items-center justify-center">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="absolute w-full h-full object-cover" 
        />
        
        {/* Overlay Canvas to allow user to visually confirm skeleton tracking */}
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="absolute w-full h-full object-cover z-0 pointer-events-none"
        />

        <div className="absolute bottom-4 left-4 bg-red-600 text-white text-xs px-2 py-1 rounded font-bold uppercase tracking-tighter shadow-md z-10 animate-pulse">
          Live Feed
        </div>

        {isRecording && (
          <div className="absolute top-4 right-4 bg-black/70 text-white text-sm px-4 py-2 rounded-full font-bold shadow-md z-10 text-center border border-white/20">
            <span className="w-2 h-2 rounded-full inline-block bg-red-500 animate-ping mr-2"></span>
            Detecting Signs...
          </div>
        )}
      </div>

      <div className="flex justify-center mt-2 gap-4">
        {!isRecording ? (
          <button 
            onClick={() => setIsRecording(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transition-all transform hover:scale-105"
          >
             Start
          </button>
        ) : (
          <button 
            onClick={() => setIsRecording(false)}
            className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transition-all transform hover:scale-105"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
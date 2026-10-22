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
  onPrediction: (text: string, probability?: number) => void;
}

export default function CameraBox({ onPrediction }: CameraProps) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  
  const lastPredictionRef = useRef({ text: "", prob: 0, timeout: null as any });
  const recentPredictionsRef = useRef<string[]>([]);
  const sequenceRef = useRef<number[][]>([]);
  const isRecordingRef = useRef(false);
  const frameCountRef = useRef(0);

  const formatPredictionText = (label: string) => {
    return label
      .replace(/_/g, " ")
      .trim()
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const getHandBoundingBox = (results: any) => {
    const points = [
      ...(results.leftHandLandmarks || []),
      ...(results.rightHandLandmarks || []),
    ];

    if (points.length === 0) return null;

    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;

    points.forEach((lm: any) => {
      minX = Math.min(minX, lm.x);
      minY = Math.min(minY, lm.y);
      maxX = Math.max(maxX, lm.x);
      maxY = Math.max(maxY, lm.y);
    });

    const padding = 0.08;
    return {
      xMin: Math.max(0, minX - padding),
      yMin: Math.max(0, minY - padding),
      xMax: Math.min(1, maxX + padding),
      yMax: Math.min(1, maxY + padding),
    };
  };

  const getCroppedHandImage = (bbox: { xMin: number; yMin: number; xMax: number; yMax: number } | null) => {
    const video = webcamRef.current?.video as HTMLVideoElement | undefined;
    if (!video || !bbox) return null;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    const sx = Math.max(0, Math.floor(bbox.xMin * vw));
    const sy = Math.max(0, Math.floor(bbox.yMin * vh));
    const ex = Math.min(vw, Math.ceil(bbox.xMax * vw));
    const ey = Math.min(vh, Math.ceil(bbox.yMax * vh));

    const sw = Math.max(1, ex - sx);
    const sh = Math.max(1, ey - sy);

    const outSize = 224;
    const canvas = document.createElement('canvas');
    canvas.width = outSize;
    canvas.height = outSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, outSize, outSize);
    return canvas.toDataURL('image/jpeg', 0.75);
  };

  useEffect(() => {
    isRecordingRef.current = isRecording;
    if (!isRecording) {
      sequenceRef.current = [];
      frameCountRef.current = 0;
      recentPredictionsRef.current = [];
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
      sequenceRef.current.push(keypoints);
      if (sequenceRef.current.length > 60) {
        sequenceRef.current.shift();
      }
      
      frameCountRef.current++;
      
      // Predict at a controlled interval (static alphabet model).
      if (frameCountRef.current % 10 === 0) {
        const bbox = getHandBoundingBox(results);
        if (!bbox) return; // no hand detected -> no request
        const croppedImage = getCroppedHandImage(bbox);
        if (!croppedImage) return;

        fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: croppedImage }),
        })
        .then(res => res.json())
        .then(data => {
            if (data.prediction && data.prediction.trim() !== '') {
                const formattedPrediction = formatPredictionText(data.prediction);
                recentPredictionsRef.current.push(formattedPrediction);
                if (recentPredictionsRef.current.length > 5) {
                    recentPredictionsRef.current.shift();
                }

                // Stability vote: require the latest prediction to appear repeatedly.
                const latestCount = recentPredictionsRef.current.filter(
                  (item) => item === formattedPrediction
                ).length;
                if (latestCount < 2) return;

                onPrediction(formattedPrediction, data.probability);
                
                // Store the prediction to draw the OpenCV-style green box
                lastPredictionRef.current.text = formattedPrediction;
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
    
    // --- VISUAL FOCUS (Hands Only) ---
    // We intentionally ignore drawing the Face and Pose to keep the webcam UI perfectly clean.
    /*
    // Draw Face
    if (results.faceLandmarks) {
      window.drawConnectors(canvasCtx, results.faceLandmarks, window.FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
    }
    // Draw Pose
    if (results.poseLandmarks) {
      window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {color: 'white', lineWidth: 2});
      window.drawLandmarks(canvasCtx, results.poseLandmarks, {color: 'red', lineWidth: 1});
    }
    */
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
            
            // 2. Format user-facing text without exposing model route details.
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
    // --- MATHEMATICAL FOCUS (Hands Only) ---
    // We force Face and Pose to be solid zeros. This blinds the AI so it doesn't 
    // get confused by your face shape or shoulder placement.
    let pose = new Array(132).fill(0);
    /* 
    if (results.poseLandmarks) {
      pose = results.poseLandmarks.map((res: any) => [res.x || 0, res.y || 0, res.z || 0, res.visibility || 0]).flat();
    }
    */
    
    let face = new Array(1404).fill(0);
    /*
    if (results.faceLandmarks) {
      face = results.faceLandmarks.slice(0, 468).map((res: any) => [res.x || 0, res.y || 0, res.z || 0]).flat();
    }
    */
    
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
      <div className="relative w-full h-[520px] bg-black rounded-3xl overflow-hidden border border-white/60 shadow-2xl flex items-center justify-center">
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

        <div className="absolute bottom-4 left-4 z-10">
          <span className="inline-flex items-center gap-2 bg-black/60 text-white text-xs px-3 py-1.5 rounded-full font-semibold shadow-md border border-white/10">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            Live
          </span>
        </div>

        {isRecording && (
          <div className="absolute top-4 right-4 bg-black/60 text-white text-sm px-4 py-2 rounded-full font-semibold shadow-md z-10 text-center border border-white/10 backdrop-blur">
            Translating...
          </div>
        )}
      </div>

      <div className="flex justify-center mt-2 gap-4">
        {!isRecording ? (
          <button 
            onClick={() => setIsRecording(true)}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold py-3 px-10 rounded-full shadow-lg transition-all transform hover:scale-[1.02] active:scale-[0.99] focus:outline-none focus:ring-4 focus:ring-blue-200"
          >
            Start
          </button>
        ) : (
          <button 
            onClick={() => setIsRecording(false)}
            className="bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-700 hover:to-red-700 text-white font-bold py-3 px-10 rounded-full shadow-lg transition-all transform hover:scale-[1.02] active:scale-[0.99] focus:outline-none focus:ring-4 focus:ring-rose-200"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
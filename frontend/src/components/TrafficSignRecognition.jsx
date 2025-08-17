// frontend/src/components/TrafficSignRecognition.jsx
import React, { useState, useRef, useEffect } from "react";

export default function TrafficSignRecognition({ apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000" }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [cameraRunning, setCameraRunning] = useState(false);
  const videoRef = useRef(null);
  const inputRef = useRef(null);

  // Handle file selection
  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setResult({ error: "Please upload an image file (png/jpg/jpeg)." });
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }

  function clearFile() {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  // Predict from uploaded file
  async function handlePredict() {
    if (!file) {
      setResult({ error: "No file selected." });
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error: ${res.status} ${text}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  // Live camera
  useEffect(() => {
    let stream;

    async function startCamera() {
      if (videoRef.current) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
          videoRef.current.srcObject = stream;
        } catch (err) {
          console.error("Cannot access camera:", err);
        }
      }
    }

    if (cameraRunning) startCamera();
    else if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [cameraRunning]);

  // Capture frame and predict
  async function captureFrame() {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      setLoading(true);
      setResult(null);

      try {
        const res = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        setResult(data);
      } catch (err) {
        setResult({ error: err.message });
      } finally {
        setLoading(false);
      }
    }, "image/jpeg");
  }

  // Auto capture while camera running
  useEffect(() => {
    if (!cameraRunning) return;
    const interval = setInterval(captureFrame, 2000); // every 2 seconds
    return () => clearInterval(interval);
  }, [cameraRunning]);

  return (
    <div className="bg-gray-900 rounded-2xl shadow-xl p-6 w-full max-w-md mx-auto text-gray-100">
      <h2 className="text-2xl font-bold mb-4 text-center">Traffic Sign Recognition</h2>

      {/* --- Upload Section --- */}
      <div className="flex flex-col items-center gap-3 mb-4">
        <label htmlFor="file" className="cursor-pointer w-full">
          <div
            className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center hover:bg-gray-800"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const droppedFile = e.dataTransfer.files?.[0];
              if (droppedFile) handleFileChange({ target: { files: [droppedFile] } });
            }}
          >
            <input
              id="file"
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <p className="text-sm text-gray-400">Click here or drop an image (png/jpg)</p>
            {preview && <p className="text-xs text-gray-300 mt-2">Selected: {file?.name}</p>}
          </div>
        </label>

        {preview && (
          <div className="w-48 h-48 bg-gray-800 rounded-md flex items-center justify-center overflow-hidden">
            <img src={preview} alt="preview" className="object-contain w-full h-full" />
          </div>
        )}

        <div className="flex gap-2 w-full">
          <button
            onClick={handlePredict}
            disabled={loading || !file}
            className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:opacity-60"
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            onClick={clearFile}
            className="flex-1 bg-gray-700 text-gray-200 py-2 rounded-lg hover:bg-gray-600"
            type="button"
          >
            Clear
          </button>
        </div>
      </div>

      {/* --- Live Camera Section --- */}
      <div className="mb-4 flex flex-col items-center gap-3">
        <video
          ref={videoRef}
          autoPlay
          className="rounded-md w-full max-w-md bg-gray-800"
        />
        <button
          onClick={() => setCameraRunning(prev => !prev)}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
        >
          {cameraRunning ? "Stop Camera" : "Start Live Recognition"}
        </button>
      </div>

      {/* --- Result Section --- */}
      {result && (
        <div className="mt-3 text-center">
          {result.error ? (
            <p className="text-sm text-red-500">{result.error}</p>
          ) : (
            <>
              <p className="text-lg font-bold">{result.class}</p>
              <p className="text-sm text-gray-300">
                Confidence: {typeof result.confidence === "number"
                  ? `${(result.confidence * 100).toFixed(2)}%`
                  : result.confidence}
              </p>
            </>
          )}
        </div>
      )}

      <div className="mt-4 text-xs text-gray-500 text-center">
        API: <span className="text-gray-400">{apiUrl}</span>
      </div>
    </div>
  );
}

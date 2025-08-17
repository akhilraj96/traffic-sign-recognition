// frontend/src/components/UploadCard.jsx
import React, { useState, useRef } from "react";

export default function UploadCard({ apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000" }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);

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

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 w-full max-w-md">
      <h2 className="text-xl font-semibold mb-4 text-center">Traffic Sign Recognition</h2>

      <div className="flex flex-col items-center gap-3">
        <label
          htmlFor="file"
          className="cursor-pointer w-full"
        >
          <div className="border-2 border-dashed rounded-lg p-4 text-center hover:bg-gray-50">
            <input
              id="file"
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <p className="text-sm text-gray-600">Click here or drop an image (png/jpg)</p>
            {preview ? <p className="text-xs text-gray-500 mt-2">Selected: {file?.name}</p> : null}
          </div>
        </label>

        {preview && (
          <div className="w-48 h-48 bg-gray-100 rounded-md flex items-center justify-center overflow-hidden">
            <img src={preview} alt="preview" className="object-contain w-full h-full" />
          </div>
        )}

        <div className="flex gap-2 w-full">
          <button
            onClick={handlePredict}
            disabled={loading}
            className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:opacity-60"
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            onClick={clearFile}
            className="flex-1 bg-gray-200 text-gray-800 py-2 rounded-lg hover:bg-gray-300"
            type="button"
          >
            Clear
          </button>
        </div>

        <div className="w-full">
          {result && result.error && (
            <div className="mt-3 text-sm text-red-600">{result.error}</div>
          )}

          {result && !result.error && (
            <div className="mt-3 text-center">
              <p className="text-lg font-bold">{result.class}</p>
              <p className="text-sm text-gray-600">
                Confidence: {typeof result.confidence === "number" ? `${(result.confidence * 100).toFixed(2)}%` : result.confidence}
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-400 text-center">
        API: <span className="text-gray-600">{apiUrl}</span>
      </div>
    </div>
  );
}

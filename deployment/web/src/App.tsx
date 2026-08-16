import { useRef, useState } from "react";
import { ApiError, predict } from "./api/client";
import { ResultCard } from "./components/ResultCard";
import type { PredictResponse } from "./types";

type Status = "idle" | "loading" | "done" | "error";

export function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File | undefined) {
    if (!file) return;

    setImageUrl(URL.createObjectURL(file));
    setStatus("loading");
    setError(null);

    try {
      const response = await predict(file);
      setResult(response);
      setStatus("done");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not reach the server.");
      setStatus("error");
    }
  }

  return (
    <main className="container">
      <h1>Plant Disease Detection</h1>
      <p className="subtitle">Upload a leaf photo to identify the crop, disease, and treatment.</p>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleFile(e.target.files?.[0])}
        style={{ display: "none" }}
      />
      <button className="upload-button" onClick={() => inputRef.current?.click()}>
        {imageUrl ? "Choose a different photo" : "Choose a photo"}
      </button>

      {imageUrl && <img className="preview" src={imageUrl} alt="Uploaded leaf" />}

      {status === "loading" && <p className="status">Analyzing...</p>}
      {status === "error" && <p className="status error">{error}</p>}
      {status === "done" && result && <ResultCard result={result} />}
    </main>
  );
}

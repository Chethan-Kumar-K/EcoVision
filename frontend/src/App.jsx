import { useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run(f, t) {
    setLoading(true);
    setError("");
    const form = new FormData();
    form.append("file", f);
    try {
      const res = await fetch(`${API}/predict?threshold=${t}`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setResult(data);
    } catch (e) {
      setResult(null);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function onFile(e) {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      run(f, threshold);
    }
  }

  // re-run when the slider is released, not on every tick
  function onRelease() {
    if (file) run(file, threshold);
  }

  return (
    <main className="page">
      <h1>EcoVision</h1>
      <p className="sub">Forest detection from Sentinel-2 imagery</p>

      <div className="controls">
        <label className="upload">
          <input type="file" accept=".tif,.tiff" onChange={onFile} />
        </label>
        <label className="slider">
          Forest threshold: {threshold.toFixed(2)}
          <input
            type="range" min="0.1" max="0.9" step="0.05" value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            onPointerUp={onRelease} onKeyUp={onRelease}
          />
        </label>
      </div>
      <p className="hint">Upload a 4-band GeoTIFF (B4, B3, B2, B8 order, raw reflectance values).</p>

      {loading && <p className="status">Running model...</p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <div className="metrics">
            <div><span>Forest cover</span><b>{result.forest_percent}%</b></div>
            <div><span>Forest area</span><b>{result.forest_ha.toLocaleString()} ha</b></div>
            <div><span>Image size</span><b>{result.width} x {result.height} px</b></div>
          </div>
          <div className="images">
            <figure><img src={result.image_png} alt="Satellite image" /><figcaption>Image (RGB)</figcaption></figure>
            <figure><img src={result.mask_png} alt="Forest mask" /><figcaption>Predicted forest (white)</figcaption></figure>
          </div>
          <p className="hint">Clouds and cloud shadows are classed as non-forest, as in the training data.</p>
        </>
      )}
    </main>
  );
}
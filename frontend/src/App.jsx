import { useRef, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState("side");
  const [opacity, setOpacity] = useState(0.55);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const latest = useRef(0);

  async function run(f, t) {
    const id = ++latest.current;
    setLoading(true);
    setError("");
    const form = new FormData();
    form.append("file", f);
    try {
      const res = await fetch(`${API}/predict?threshold=${t}`, { method: "POST", body: form });
      const data = await res.json();
      if (id !== latest.current) return;   // a newer request replaced this one
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setResult(data);
    } catch (e) {
      if (id !== latest.current) return;
      setResult(null);
      setError(e.message === "Failed to fetch" ? "Cannot reach the API. Is the backend running on port 8000?" : e.message);
    } finally {
      if (id === latest.current) setLoading(false);
    }
  }

  function pick(f) {
    if (!f) return;
    if (!/\.tiff?$/i.test(f.name)) {
      setError("Please choose a .tif or .tiff file.");
      return;
    }
    setFile(f);
    run(f, threshold);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files[0]);
  }

  function onRelease() {
    if (file) run(file, threshold);
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">EcoVision</div>
        <span className="tag">Forest detection from Sentinel-2</span>
      </header>

      <main className="page">
        <section
          className={`dropzone ${dragging ? "drag" : ""}`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef} type="file" accept=".tif,.tiff" hidden
            onChange={(e) => pick(e.target.files[0])}
          />
          <p className="drop-title">{file ? file.name : "Drop a GeoTIFF here, or click to choose"}</p>
          <p className="hint">4 bands in the order B4, B3, B2, B8, raw L2A reflectance values</p>
        </section>

        <section className="panel">
          <label className="slider">
            <span>Forest threshold <b>{threshold.toFixed(2)}</b></span>
            <input
              type="range" min="0.1" max="0.9" step="0.05" value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              onPointerUp={onRelease} onKeyUp={onRelease}
            />
          </label>
          <p className="hint">Lower finds more forest, higher is stricter. Updates when you release the slider.</p>
        </section>

        {error && <div className="error">{error}</div>}
        {loading && <div className="loading"><span className="spinner" /> Running model...</div>}

        {result && (
          <section className={loading ? "results dim" : "results"}>
            <div className="metrics">
              <div className="card wide">
                <span>Forest cover</span>
                <b>{result.forest_percent}%</b>
                <div className="bar"><div style={{ width: `${result.forest_percent}%` }} /></div>
              </div>
              <div className="card">
                <span>Forest area</span>
                <b>{result.forest_ha.toLocaleString()} ha</b>
              </div>
              <div className="card">
                <span>Image size</span>
                <b>{result.width} x {result.height}</b>
              </div>
            </div>

            <div className="toolbar">
              <div className="segmented">
                <button className={view === "side" ? "on" : ""} onClick={() => setView("side")}>Side by side</button>
                <button className={view === "overlay" ? "on" : ""} onClick={() => setView("overlay")}>Overlay</button>
              </div>
              {view === "overlay" && (
                <label className="slider inline">
                  <span>Overlay opacity</span>
                  <input type="range" min="0.1" max="1" step="0.05" value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))} />
                </label>
              )}
            </div>

            {view === "side" ? (
              <div className="images">
                <figure><img src={result.image_png} alt="Satellite image" /><figcaption>Image (RGB)</figcaption></figure>
                <figure><img src={result.mask_png} alt="Predicted forest mask" /><figcaption>Predicted forest (white)</figcaption></figure>
              </div>
            ) : (
              <figure className="single">
                <div className="stack">
                  <img src={result.image_png} alt="Satellite image" />
                  <img className="top" src={result.overlay_png} alt="Forest overlay" style={{ opacity }} />
                </div>
                <figcaption>Predicted forest in green over the image</figcaption>
              </figure>
            )}

            <p className="hint">Clouds and cloud shadows are classed as non-forest, as in the training data.</p>
          </section>
        )}
      </main>
    </div>
  );
}
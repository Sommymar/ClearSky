import { useState, useCallback, useRef } from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface Result {
  original_size: { width: number; height: number };
  streak_coverage: number;
  detected: boolean;
  input: string;
  overlay: string;
  cleaned: string;
  mask: string;
}

type Status = "idle" | "uploading" | "done" | "error";

export default function StreakStudio() {
  const [result, setResult] = useState<Result | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const process = useCallback(async (file: File) => {
    setStatus("uploading");
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch(`${API}/api/process`, { method: "POST", body });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Processing failed");
      setResult(await res.json());
      setStatus("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setStatus("error");
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) process(file);
    },
    [process]
  );

  return (
    <div className="studio">
      <header className="studio__head">
        <h1>ClearSky</h1>
        <p>Detect and remove satellite trails from telescope images.</p>
      </header>

      <div
        className={`dropzone${dragging ? " dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/tiff"
          hidden
          onChange={(e) => e.target.files?.[0] && process(e.target.files[0])}
        />
        {status === "uploading" ? (
          <p>Processing your image…</p>
        ) : (
          <p>Drop a telescope image here, or click to choose one</p>
        )}
      </div>

      {status === "error" && <p className="error">{error}</p>}

      {status === "done" && result && (
        <section className="results">
          <div className="verdict">
            {result.detected ? (
              <span className="verdict__flag">
                Streak detected — {(result.streak_coverage * 100).toFixed(1)}% of frame affected
              </span>
            ) : (
              <span className="verdict__clean">No significant streak found</span>
            )}
          </div>
          <div className="panels">
            <Panel label="Input" src={result.input} />
            <Panel label="Detected streak" src={result.overlay} />
            <Panel label="Cleaned" src={result.cleaned} />
          </div>
        </section>
      )}
    </div>
  );
}

function Panel({ label, src }: { label: string; src: string }) {
  return (
    <figure className="panel">
      <img src={src} alt={label} />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

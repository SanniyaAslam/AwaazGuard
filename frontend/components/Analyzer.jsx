"use client";
import { useEffect, useRef, useState } from "react";

const MAX_RECORD_S = 30;

const VERDICTS = {
  safe: {
    title: "This sounds like a real person",
    body: "We found no strong signs of an AI-generated voice. Scammers can still use a real voice, so verify unusual requests another way.",
  },
  suspicious: {
    title: "We can't tell if this voice is real",
    body: "Some parts look machine-made. Treat this message with caution and confirm it by calling the person on a number you already know.",
  },
  likely_fake: {
    title: "This voice is likely AI-generated",
    body: "The recording has strong signs of a cloned or synthetic voice. Do not send money or codes. Call the person back on a number you already know.",
  },
};

function tone(p) {
  if (p >= 0.65) return "var(--fake)";
  if (p >= 0.35) return "var(--warn)";
  return "var(--safe)";
}

function Waveform({ file, segments }) {
  const waveRef = useRef(null);
  const specRef = useRef(null);
  const wsRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    let ws;
    let cancelled = false;
    (async () => {
      const WaveSurfer = (await import("wavesurfer.js")).default;
      const Regions = (await import("wavesurfer.js/dist/plugins/regions.esm.js")).default;
      const Spectrogram = (await import("wavesurfer.js/dist/plugins/spectrogram.esm.js")).default;
      if (cancelled) return;
      const css = getComputedStyle(document.documentElement);
      const regions = Regions.create();
      ws = WaveSurfer.create({
        container: waveRef.current,
        height: 96,
        waveColor: css.getPropertyValue("--wave").trim(),
        progressColor: css.getPropertyValue("--ink").trim(),
        cursorColor: css.getPropertyValue("--ink").trim(),
        barWidth: 2, barGap: 1, barRadius: 2,
        plugins: [regions, Spectrogram.create({ container: specRef.current, height: 110, labels: false })],
      });
      wsRef.current = ws;
      ws.on("decode", () => {
        segments.forEach((s) => {
          if (s.prob_fake >= 0.35) {
            regions.addRegion({
              start: s.start, end: s.end, drag: false, resize: false,
              color: s.prob_fake >= 0.65 ? "rgba(184,33,58,0.22)" : "rgba(185,133,0,0.20)",
            });
          }
        });
      });
      ws.on("play", () => setPlaying(true));
      ws.on("pause", () => setPlaying(false));
      ws.on("finish", () => setPlaying(false));
      ws.loadBlob(file);
    })();
    return () => { cancelled = true; ws && ws.destroy(); };
  }, [file, segments]);

  return (
    <>
      <div className="wave-wrap">
        <div ref={waveRef} />
        <div className="heat" aria-hidden="true">
          {segments.map((s) => <div key={s.start} style={{ background: tone(s.prob_fake) }} title={`${s.start}-${s.end}s`} />)}
        </div>
        <div ref={specRef} />
      </div>
      <div className="player">
        <button className="btn" onClick={() => wsRef.current?.playPause()}>{playing ? "Pause" : "Play recording"}</button>
        <div className="legend">
          <span><i style={{ background: "var(--safe)" }} />Sounds human</span>
          <span><i style={{ background: "var(--warn)" }} />Unclear</span>
          <span><i style={{ background: "var(--fake)" }} />Sounds machine-made</span>
        </div>
      </div>
    </>
  );
}

function Feedback({ id }) {
  const [sent, setSent] = useState(null);
  const send = async (label) => {
    const r = await fetch("/api/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prediction_id: id, label }),
    });
    setSent(r.ok ? label : "error");
  };
  if (sent === "error") return <div className="feedback"><p className="error">Feedback could not be saved. Try again later.</p></div>;
  if (sent) return <div className="feedback"><p className="thanks">Thanks. Your answer will help train the next model.</p></div>;
  return (
    <div className="feedback">
      <p>Do you know who really made this recording?</p>
      <button className="btn" onClick={() => send("bonafide")}>A real person</button>
      <button className="btn" onClick={() => send("spoof")}>An AI voice</button>
    </div>
  );
}

export default function Analyzer() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [over, setOver] = useState(false);
  const [rec, setRec] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const inputRef = useRef(null);

  async function analyze(f) {
    setFile(f); setResult(null); setError(""); setBusy(true);
    try {
      const body = new FormData();
      body.append("file", f, f.name || "recording.webm");
      const r = await fetch("/api/predict", { method: "POST", body });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `The check failed (HTTP ${r.status}).`);
      setResult(data);
    } catch (e) {
      setError(e.message === "Failed to fetch" ? "The checking service is not reachable. Make sure the gateway is running." : e.message);
    } finally {
      setBusy(false);
    }
  }

  async function toggleRecord() {
    if (rec) { rec.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      const chunks = [];
      let t = 0;
      const timer = setInterval(() => { t += 1; setElapsed(t); if (t >= MAX_RECORD_S) mr.stop(); }, 1000);
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = () => {
        clearInterval(timer); stream.getTracks().forEach((tr) => tr.stop());
        setRec(null); setElapsed(0);
        const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
        analyze(new File([blob], "recording.webm", { type: blob.type }));
      };
      mr.start(); setRec(mr); setElapsed(0);
    } catch {
      setError("Microphone access was blocked. Allow it in your browser settings, or upload a file instead.");
    }
  }

  const v = result && VERDICTS[result.verdict];
  const pct = result ? Math.round(result.prob_fake * 100) : 0;

  return (
    <>
      <section className="intro">
        <h1>Is this voice real?</h1>
        <p>Upload a voice note or call recording, or record one now. AwaazGuard checks whether it was spoken by a person or generated by an AI voice clone, and shows which seconds look suspicious.</p>
      </section>

      <section className="capture" aria-label="Add a recording">
        <label
          className={`drop${over ? " over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); e.dataTransfer.files[0] && analyze(e.dataTransfer.files[0]); }}
        >
          <input ref={inputRef} type="file" accept="audio/*,.opus,.ogg,.m4a,.webm"
                 onChange={(e) => e.target.files[0] && analyze(e.target.files[0])} />
          <strong>{file && !rec ? file.name : "Choose a recording or drop it here"}</strong>
          <span>WhatsApp voice notes (.opus), MP3, WAV, M4A or OGG. The first 30 seconds are checked.</span>
        </label>
        <button className={`record${rec ? " on" : ""}`} onClick={toggleRecord} disabled={busy}>
          <span className="dot" />
          {rec ? `Stop (${elapsed}s)` : "Record with mic"}
          <small>{rec ? `Max ${MAX_RECORD_S} seconds` : "Hold your phone near the speaker"}</small>
        </button>
      </section>

      <p className="status" role="status">
        {busy && "Checking the recording..."}
        {error && <span className="error">{error}</span>}
      </p>

      {result && v && (
        <section className={`result tone-${result.verdict}`} aria-live="polite">
          <div className="verdict">
            <h2>{v.title}</h2>
            <p>{v.body}</p>
            <div className="meter" aria-label={`Fake likelihood ${pct} percent`}>
              <div className="meter-track"><div className="meter-pin" style={{ left: `${pct}%` }} /></div>
              <div className="meter-scale"><span>Human</span><span>{pct}% likely AI-generated</span><span>Machine-made</span></div>
            </div>
          </div>
          <div className="evidence">
            <h3>Where it sounds machine-made</h3>
            <p>Each second of the recording is scored separately. Highlighted parts drove the verdict.</p>
            <Waveform file={file} segments={result.segments} />
          </div>
          <dl className="facts">
            <div><dt>Length</dt><dd>{result.duration_s.toFixed(1)} s</dd></div>
            <div><dt>Checked in</dt><dd>{(result.latency_ms / 1000).toFixed(1)} s</dd></div>
            <div><dt>Model</dt><dd>v{result.model_version} ({result.variant})</dd></div>
            <div><dt>Check ID</dt><dd>#{result.id}</dd></div>
          </dl>
          <Feedback key={result.id} id={result.id} />
        </section>
      )}

      <section className="howto" aria-label="How it works">
        <div><h3>Scored second by second</h3><p>The recording is cut into overlapping 4-second windows, so a short cloned sentence inside a real call still stands out.</p></div>
        <div><h3>Built for real calls</h3><p>The model is trained on phone-line, WhatsApp and compressed audio, not only clean studio recordings.</p></div>
        <div><h3>Tested against new fakes</h3><p>Every new model version must catch freshly generated voice clones before it is released.</p></div>
      </section>
    </>
  );
}

import React, { useState } from "react";

function AudioInput({ onPredict }) {
  const [file, setFile] = useState(null);

  return (
    <div className="card">
      <h2>Upload or Record Audio</h2>

      <input
        type="file"
        accept=".wav,.mp3"
        onChange={(e) => setFile(e.target.files[0])}
      />

      {file && <p>🎧 {file.name}</p>}

      <button onClick={onPredict} disabled={!file}>
        Recognize Instrument
      </button>

      <button style={{ marginTop: "10px" }} disabled>
        🎙 Record Audio (Coming Soon)
      </button>
    </div>
  );
}

export default AudioInput;

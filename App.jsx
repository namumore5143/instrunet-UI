import React, { useState } from "react";
import Navbar from "./Navbar";
import Sidebar from "./SideBar";
import AudioInput from "./AudioInput";
import ResultPanel from "./ResultPanel";
import Loader from "./Loader";
import "./style.css";

function App() {
  const [loading, setLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [activeSection, setActiveSection] = useState("upload");

  const simulatePrediction = () => {
    setLoading(true);
    setShowResult(false);

    setTimeout(() => {
      setLoading(false);
      setShowResult(true);
    }, 2000);
  };

  return (
    <>
      <Navbar />

      <div className="dashboard">
        <Sidebar onSelect={setActiveSection} />

       <main className="main-content">
  {activeSection === "upload" && (
    <div className="analysis-container">   {/* 👈 ADD THIS */}
      <AudioInput onPredict={simulatePrediction} />
      {loading && <Loader />}
      {showResult && <ResultPanel />}
    </div>
  )}


          {activeSection === "history" && (
            <div className="card">
              <h2>🎧 Previous Analyses</h2>
              <p>No previous data available.</p>
            </div>
          )}

          {activeSection === "visuals" && (
            <div className="card">
              <h2>📊 Visualizations</h2>
              <p>Waveform & spectrogram will appear here.</p>
            </div>
          )}

          {activeSection === "reports" && (
            <div className="card">
              <h2>📄 Download Reports</h2>
              <button>Download JSON</button>
              <button style={{ marginTop: "10px" }}>Download PDF</button>
            </div>
          )}
        </main>
      </div>

     
    </>
  );
}

export default App;

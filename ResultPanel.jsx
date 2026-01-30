import React from "react";
function ResultPanel() {
  const results = [
    { name: "Guitar", value: 93 },
    { name: "Violin", value: 4 },
    { name: "Piano", value: 3 },
  ];

  return (
    <div className="card">
      <h2>Prediction Result</h2>

      {results.map((item) => (
        <div key={item.name} className="bar-row">
          <span>{item.name}</span>
          <div className="bar">
            <div
              className="bar-fill"
              style={{ width: `${item.value}%` }}
            ></div>
          </div>
          <span>{item.value}%</span>
        </div>
      ))}
    </div>
  );
}

export default ResultPanel;

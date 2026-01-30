import React from "react";

function Sidebar({ onSelect }) {
  return (
    <aside className="sidebar">
      <h3 className="sidebar-title">Menu</h3>

      <ul className="sidebar-menu">
        <li onClick={() => onSelect("upload")}>📤 Upload Audio</li>
        <li onClick={() => onSelect("history")}>🎧 Previous Analysis</li>
        <li onClick={() => onSelect("visuals")}>📊 Visualizations</li>
        <li onClick={() => onSelect("reports")}>📄 Download Reports</li>
      </ul>
    </aside>
  );
}

export default Sidebar;

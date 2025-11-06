"use client";
import { useState } from "react";
import EEGChart from "./Chart";

export interface Prediction {
  created_at: string;
  raw_data: number[][];
  confidence: number;
  predicted_class: number;
}

export default function EEGHistoryViewer({
  predictions,
}: {
  predictions: Prediction[];
}) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  if (!predictions || predictions.length === 0) {
    return <p className="text-gray-400 italic">No login history available</p>;
  }

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedIndex(parseInt(e.target.value));
  };

  const selected = predictions[selectedIndex];

  return (
    <div className="flex flex-col gap-6 w-full">
      <div>
        <label htmlFor="eegSelect" className="block text-sm font-medium mb-2">
          🧠 Select EEG login record:
        </label>
        <select
          id="eegSelect"
          value={selectedIndex}
          onChange={handleChange}
          className="border border-gray-300 rounded-md px-3 py-2 w-full text-sm"
        >
          {predictions.map((p, i) => (
            <option key={i} value={i}>
              {new Date(p.created_at).toLocaleString()} — Class {p.predicted_class} (
              {(p.confidence * 100).toFixed(2)}%)
            </option>
          ))}
        </select>
      </div>

      {/* EEG visualization */}
      <div className="mt-4">
        <h2 className="text-lg font-semibold mb-2">
          Selected: {new Date(selected.created_at).toLocaleString()}
        </h2>
        <EEGChart
          rawData={selected.raw_data}
          channelNames={["Iz", "O2", "Oz"]}
        />
      </div>
    </div>
  );
}

"use client";
import { useEffect, useState } from "react";
import EEGChart from "../components/Chart";

export default function EEGDashboardClient() {
  const [rawData, setRawData] = useState<number[][]>([]);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [predictedClass, setPredictedClass] = useState<number | null>(null);

  useEffect(() => {
    const savedData = localStorage.getItem("eeg_raw_data");
    const conf = localStorage.getItem("eeg_confidence");
    const clazz = localStorage.getItem("eeg_class");

    if (savedData) setRawData(JSON.parse(savedData));
    if (conf) setConfidence(parseFloat(conf));
    if (clazz) setPredictedClass(parseInt(clazz));
  }, []);

  if (!rawData.length) {
    return <p className="text-gray-400 italic">No EEG data available</p>;
  }

  return (
    <div className="flex-1">
      <h2 className="text-lg font-bold mb-6">
        MODEL CONFIDENCE:{" "}
        <span className="text-black">
          {confidence ? `${(confidence * 100).toFixed(2)}%` : "N/A"}
        </span>{" "}
        (Class: {predictedClass ?? "?"})
      </h2>

      <EEGChart rawData={rawData} channelNames={["Iz", "O2", "Oz"]} />
    </div>
  );
}

"use client";
import EEGChart from "../components/Chart";
interface EEGDashboardClientProps {
  rawData: number[][];
  confidence: number | null;
  predictedClass: number | null;
}

export default function EEGDashboardClient({
  rawData,
  confidence,
  predictedClass,
}: EEGDashboardClientProps) {
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

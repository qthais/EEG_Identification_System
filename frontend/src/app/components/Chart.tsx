"use client";

import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface EEGChartProps {
  rawData: number[][];        // from API: [[ch1, ch2, ch3], ...]
  channelNames?: string[];    // optional: e.g. ["OZ", "IZ", "PZ"]
}

export default function EEGChart({ rawData, channelNames }: EEGChartProps) {
  if (!rawData || rawData.length === 0) {
    return <p className="text-gray-400 italic">No EEG data available</p>;
  }

  // Dynamically convert rawData into Recharts-friendly array
  const data = rawData.map((sample, i) => {
    const point: any = { x: i }; // sample index
    sample.forEach((val, chIndex) => {
      const key = channelNames?.[chIndex] || `Ch${chIndex + 1}`;
      point[key] = val;
    });
    return point;
  });

  // Auto-generate colors for each channel
  const colors = ["#facc15", "#818cf8", "#34d399", "#f87171", "#a855f7"];

  return (
    <div className="w-full h-[350px] p-4 bg-white rounded-2xl shadow">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
          <XAxis dataKey="x" label={{ value: "Time (samples)", position: "insideBottom", offset: -5 }} />
          <YAxis label={{ value: "Amplitude", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />

          {Object.keys(data[0])
            .filter((k) => k !== "x")
            .map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={colors[i % colors.length]}
                strokeWidth={2}
                dot={false}
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

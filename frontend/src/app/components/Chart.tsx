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

const data = [
  { x: 0, oz: -500, iz: -700 },
  { x: 0.5, oz: 1000, iz: -200 },
  { x: 1, oz: 800, iz: 0 },
  { x: 1.5, oz: 400, iz: -300 },
  { x: 2, oz: 600, iz: -400 },
  { x: 2.5, oz: 300, iz: -200 },
  { x: 3, oz: 900, iz: 100 },
];

export default function EEGChart() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid stroke="#e5e7eb" />
        <XAxis dataKey="x" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="oz" stroke="#facc15" strokeWidth={2} />
        <Line type="monotone" dataKey="iz" stroke="#818cf8" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

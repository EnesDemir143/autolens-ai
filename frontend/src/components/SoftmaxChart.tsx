// ── SoftmaxChart — Recharts horizontal bar, top-3 highlighted ───────────────
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "../store";

// Color tiers
const RANK_COLORS = [
  "var(--autolens-accent)",     // #1 — bright
  "var(--autolens-accent-dim)", // #2 — medium
  "#3d4f60",                    // #3 — subtle blue-gray
];
const MUTED_COLOR = "#252930";  // rank 4–8
const RANK_LABELS = ["#1", "#2", "#3"];

export function SoftmaxChart() {
  const { result, status } = useStore();

  if (status !== "done" || !result) return null;

  const data = Object.entries(result.probabilities)
    .map(([label, prob]) => ({ label, prob }))
    .sort((a, b) => b.prob - a.prob);

  return (
    <AnimatePresence>
      <motion.div
        key="softmax"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
        style={{
          background: "var(--autolens-surface)",
          border: "1px solid var(--autolens-border)",
          borderRadius: "12px",
          padding: "1rem 1.25rem",
        }}
      >
        <div
          style={{
            fontSize: "0.65rem",
            color: "var(--autolens-text-muted)",
            letterSpacing: "0.12em",
            marginBottom: "0.75rem",
          }}
        >
          CLASS PROBABILITIES
        </div>
        <ResponsiveContainer width="100%" height={data.length * 32 + 10}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 0, right: 52, bottom: 0, left: 0 }}
            barCategoryGap="25%"
          >
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "var(--autolens-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={128}
              axisLine={false}
              tickLine={false}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              tick={((props: any) => {
                const { x, y, payload, index } = props as { x: number; y: number; payload: { value: string }; index: number };
                const isTop3 = index < 3;
                return (
                  <g transform={`translate(${x},${y})`}>
                    {isTop3 && (
                      <text
                        x={-128}
                        y={0}
                        dy={4}
                        textAnchor="start"
                        fill={RANK_COLORS[index]}
                        fontSize={9}
                        fontFamily="var(--font-mono)"
                        fontWeight={700}
                      >
                        {RANK_LABELS[index]}
                      </text>
                    )}
                    <text
                      x={-104}
                      y={0}
                      dy={4}
                      textAnchor="start"
                      fill={isTop3 ? "var(--autolens-text-dim)" : "var(--autolens-text-muted)"}
                      fontSize={11}
                      fontFamily="var(--font-mono)"
                    >
                      {payload.value}
                    </text>
                  </g>
                );
              }) as any}
            />
            <Tooltip
              formatter={(v) => [`${((v as number) * 100).toFixed(2)}%`, "Probability"]}
              contentStyle={{
                background: "var(--autolens-surface-2)",
                border: "1px solid var(--autolens-border)",
                borderRadius: "6px",
                fontFamily: "var(--font-mono)",
                fontSize: "0.75rem",
                color: "var(--autolens-text)",
              }}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Bar dataKey="prob" radius={[0, 3, 3, 0]} isAnimationActive={true}>
              {data.map((entry, index) => (
                <Cell
                  key={entry.label}
                  fill={index < 3 ? RANK_COLORS[index] : MUTED_COLOR}
                  fillOpacity={index >= 3 ? 0.5 : 1}
                />
              ))}
              <LabelList
                dataKey="prob"
                position="right"
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={((v: number) => `${(v * 100).toFixed(1)}%`) as any}
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "10px",
                  fill: "var(--autolens-text-dim)",
                }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>
    </AnimatePresence>
  );
}

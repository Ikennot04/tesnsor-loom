import type { NetworkSnapshot } from "../types/type";

interface Props {
  snapshot: NetworkSnapshot | null;
}

const MIN_WIDTH = 640;
const LAYER_GAP = 140;
const PAD_X = 70;
const PAD_Y = 40;
const MIN_HEIGHT = 360;
const NODE_SLOT = 20;

const COLOR = {
  surface: "#F8FAFC",
  secondary: "#4A5568",
  primary: "#0A192F",
  accent: "#10B981",
  negative: "#EF4444",
  node: "#0A192F",
};

function layerName(layer: number, layerCount: number): string {
  if (layer === 0) return "Input";
  if (layer === layerCount - 1) return "Output";
  return `Hidden ${layer}`;
}

function layerX(layer: number, layerCount: number, width: number): number {
  if (layerCount <= 1) return width / 2;
  return PAD_X + (layer * (width - 2 * PAD_X)) / (layerCount - 1);
}

function nodeY(index: number, count: number, height: number): number {
  return PAD_Y + ((index + 0.5) * (height - 2 * PAD_Y)) / count;
}

function maxAbs(values: number[]): number {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m > 0 ? m : 1;
}

/** Live network drawing: edge width/opacity = |weight|, green = positive, red = negative,
 *  node opacity = mean activation. Draws every node in every layer, for any number of layers. */
export function NetworkVisualizer({ snapshot }: Props) {
  if (!snapshot) {
    return (
      <p className="text-sm text-secondary">
        The network appears here once training starts.
      </p>
    );
  }

  const sizes = snapshot.layer_sizes;
  const layerCount = sizes.length;
  const biggest = Math.max(...sizes, 1);

  const width = Math.max(MIN_WIDTH, 2 * PAD_X + (layerCount - 1) * LAYER_GAP);
  const height = Math.max(MIN_HEIGHT, biggest * NODE_SLOT + 2 * PAD_Y);
  const slot = (height - 2 * PAD_Y) / biggest;
  const radius = Math.max(1.5, Math.min(9, slot / 2 - 0.5));

  const edgeMax = maxAbs(snapshot.edges.flatMap((block) => block.weights));

  return (
    <figure className="m-0 overflow-x-auto">
      <svg
        className="block max-w-none"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Neural network at epoch ${snapshot.epoch}`}
      >
        <rect
          x={0}
          y={0}
          width={width}
          height={height}
          fill={COLOR.surface}
          stroke={COLOR.secondary}
          strokeOpacity={0.35}
        />

        {snapshot.edges.map((block, k) => {
          const fromCount = sizes[k];
          const toCount = sizes[k + 1];
          const lines = [];
          for (let r = 0; r < block.rows; r++) {
            for (let c = 0; c < block.cols; c++) {
              const w = block.weights[r * block.cols + c];
              const strength = Math.abs(w) / edgeMax;
              lines.push(
                <line
                  key={`${k}-${r}-${c}`}
                  x1={layerX(k, layerCount, width)}
                  y1={nodeY(r, fromCount, height)}
                  x2={layerX(k + 1, layerCount, width)}
                  y2={nodeY(c, toCount, height)}
                  stroke={w >= 0 ? COLOR.accent : COLOR.negative}
                  strokeWidth={0.3 + 2.2 * strength}
                  strokeOpacity={0.12 + 0.8 * strength}
                />,
              );
            }
          }
          return <g key={`edges-${k}`}>{lines}</g>;
        })}

        {sizes.map((count, l) => {
          const activity = snapshot.node_activity[l] ?? [];
          const layerMax = maxAbs(activity);
          return (
            <g key={`layer-${l}`}>
              {Array.from({ length: count }, (_, i) => {
                const level = Math.abs(activity[i] ?? 0) / layerMax;
                return (
                  <circle
                    key={i}
                    cx={layerX(l, layerCount, width)}
                    cy={nodeY(i, count, height)}
                    r={radius}
                    fill={COLOR.node}
                    fillOpacity={0.2 + 0.8 * level}
                    stroke={COLOR.accent}
                    strokeWidth={radius > 4 ? 1.25 : 0.75}
                  />
                );
              })}
              <text
                x={layerX(l, layerCount, width)}
                y={height - 12}
                textAnchor="middle"
                fontSize={12}
                fill={COLOR.secondary}
                fontFamily="Plus Jakarta Sans, sans-serif"
                fontWeight={600}
              >
                {layerName(l, layerCount) + " (" + count + ")"}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-2 text-xs text-secondary">
        Epoch {snapshot.epoch}. Green = positive weight, red = negative, thicker = larger.
        Node brightness = mean activation.
      </figcaption>
    </figure>
  );
}

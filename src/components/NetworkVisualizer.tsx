import type { NetworkSnapshot } from "../types/type";

interface Props {
  snapshot: NetworkSnapshot | null;
}

const WIDTH = 640;
const PAD_X = 70;
const PAD_Y = 40;
const MIN_HEIGHT = 360;
const NODE_SLOT = 20; // vertical space reserved per node in the biggest layer
const LAYER_NAMES = ["Input", "Hidden", "Output"];

function layerX(layer: number, layerCount: number): number {
  if (layerCount <= 1) return WIDTH / 2;
  return PAD_X + (layer * (WIDTH - 2 * PAD_X)) / (layerCount - 1);
}

function nodeY(index: number, count: number, height: number): number {
  return PAD_Y + ((index + 0.5) * (height - 2 * PAD_Y)) / count;
}

function maxAbs(values: number[]): number {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m > 0 ? m : 1;
}

/** Live network drawing: edge width/opacity = |weight|, blue = positive, red = negative,
 *  node opacity = mean activation. Draws every node in every layer. */
export function NetworkVisualizer({ snapshot }: Props) {
  if (!snapshot) {
    return <p>The network appears here once training starts.</p>;
  }

  const sizes = snapshot.layer_sizes;
  const layerCount = sizes.length;
  const biggest = Math.max(...sizes, 1);
  const height = Math.max(MIN_HEIGHT, biggest * NODE_SLOT + 2 * PAD_Y);
  const slot = (height - 2 * PAD_Y) / biggest;
  const radius = Math.max(1.5, Math.min(9, slot / 2 - 0.5));
  const edgeMax = maxAbs(snapshot.edges.flatMap((block) => block.weights));

  return (
    <figure>
      <svg
        width={WIDTH}
        height={height}
        viewBox={`0 0 ${WIDTH} ${height}`}
        role="img"
        aria-label={`Neural network at epoch ${snapshot.epoch}`}
      >
        <rect x={0} y={0} width={WIDTH} height={height} fill="white" stroke="gray" />

        {/* edges first so nodes draw on top */}
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
                  x1={layerX(k, layerCount)}
                  y1={nodeY(r, fromCount, height)}
                  x2={layerX(k + 1, layerCount)}
                  y2={nodeY(c, toCount, height)}
                  stroke={w >= 0 ? "steelblue" : "tomato"}
                  strokeWidth={0.3 + 2.2 * strength}
                  strokeOpacity={0.12 + 0.8 * strength}
                />,
              );
            }
          }
          return <g key={`edges-${k}`}>{lines}</g>;
        })}

        {/* nodes */}
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
                    cx={layerX(l, layerCount)}
                    cy={nodeY(i, count, height)}
                    r={radius}
                    fill="goldenrod"
                    fillOpacity={0.15 + 0.85 * level}
                    stroke="black"
                    strokeWidth={radius > 4 ? 1 : 0.5}
                  />
                );
              })}
              <text x={layerX(l, layerCount)} y={height - 12} textAnchor="middle" fontSize={12}>
                {(LAYER_NAMES[l] ?? `Layer ${l}`) + " (" + count + ")"}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        Epoch {snapshot.epoch}. Blue = positive weight, red = negative, thicker = larger. Node
        brightness = mean activation.
      </figcaption>
    </figure>
  );
}
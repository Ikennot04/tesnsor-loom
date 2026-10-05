import type { NetworkSnapshot } from "../types/type";

interface Props {
  snapshot: NetworkSnapshot | null;
}

const WIDTH = 640;
const HEIGHT = 360;
const PAD_X = 70;
const PAD_Y = 40;
const LAYER_NAMES = ["Input", "Hidden", "Output"];

function layerX(layer: number, layerCount: number): number {
  if (layerCount <= 1) return WIDTH / 2;
  return PAD_X + (layer * (WIDTH - 2 * PAD_X)) / (layerCount - 1);
}

function nodeY(index: number, count: number): number {
  return PAD_Y + ((index + 0.5) * (HEIGHT - 2 * PAD_Y)) / count;
}

function maxAbs(values: number[]): number {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m > 0 ? m : 1;
}

/** Live network drawing: edge width/opacity = |weight|, blue = positive, red = negative,
 *  node opacity = mean activation. Uses SVG attributes only (no CSS). */
export function NetworkVisualizer({ snapshot }: Props) {
  if (!snapshot) {
    return <p>The network appears here once training starts.</p>;
  }

  const layerCount = snapshot.shown_sizes.length;
  const edgeMax = maxAbs(snapshot.edges.flatMap((block) => block.weights));

  return (
    <figure>
      <svg
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Neural network at epoch ${snapshot.epoch}`}
        // border="1"
      >
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="white" stroke="gray" />

        {/* edges first so nodes draw on top */}
        {snapshot.edges.map((block, k) => {
          const fromCount = snapshot.shown_sizes[k];
          const toCount = snapshot.shown_sizes[k + 1];
          const lines = [];
          for (let r = 0; r < block.rows; r++) {
            for (let c = 0; c < block.cols; c++) {
              const w = block.weights[r * block.cols + c];
              const strength = Math.abs(w) / edgeMax;
              lines.push(
                <line
                  key={`${k}-${r}-${c}`}
                  x1={layerX(k, layerCount)}
                  y1={nodeY(r, fromCount)}
                  x2={layerX(k + 1, layerCount)}
                  y2={nodeY(c, toCount)}
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
        {snapshot.shown_sizes.map((count, l) => {
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
                    cy={nodeY(i, count)}
                    r={9}
                    fill="goldenrod"
                    fillOpacity={0.15 + 0.85 * level}
                    stroke="black"
                    strokeWidth={1}
                  />
                );
              })}
              <text x={layerX(l, layerCount)} y={HEIGHT - 12} textAnchor="middle" fontSize={12}>
                {(LAYER_NAMES[l] ?? `Layer ${l}`) + " (" + snapshot.layer_sizes[l] + ")"}
              </text>
              {snapshot.layer_sizes[l] > count && (
                <text x={layerX(l, layerCount)} y={20} textAnchor="middle" fontSize={10}>
                  {`showing ${count} of ${snapshot.layer_sizes[l]}`}
                </text>
              )}
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
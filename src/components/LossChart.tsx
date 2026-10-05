interface Props {
  values: number[];
}

const WIDTH = 640;
const HEIGHT = 140;
const PAD = 24;

/** Minimal loss-over-epochs line chart (SVG attributes only). */
export function LossChart({ values }: Props) {
  if (values.length < 2) return <p>Loss chart appears after the second epoch.</p>;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;

  const points = values
    .map((v, i) => {
      const x = PAD + (i * (WIDTH - 2 * PAD)) / (values.length - 1);
      const y = HEIGHT - PAD - ((v - min) / span) * (HEIGHT - 2 * PAD);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <figure>
      <svg width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Loss per epoch">
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="white" stroke="gray" />
        <polyline points={points} fill="none" stroke="steelblue" strokeWidth={2} />
        <text x={PAD} y={14} fontSize={11}>{`max ${max.toPrecision(4)}`}</text>
        <text x={PAD} y={HEIGHT - 6} fontSize={11}>{`min ${min.toPrecision(4)}`}</text>
      </svg>
      <figcaption>Loss per epoch</figcaption>
    </figure>
  );
}
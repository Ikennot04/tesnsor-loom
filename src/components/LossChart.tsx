interface Props {
  values: number[];
}

const WIDTH = 640;
const HEIGHT = 140;
const PAD = 24;

/** Minimal loss-over-epochs line chart (SVG attributes only). */
export function LossChart({ values }: Props) {
  if (values.length < 2) {
    return (
      <p className="text-sm text-secondary">
        Loss chart appears after the second epoch.
      </p>
    );
  }

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
    <figure className="m-0">
      <svg
        className="block h-auto w-full max-w-full"
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Loss per epoch"
      >
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="#F8FAFC" stroke="#4A5568" strokeOpacity={0.35} />
        <polyline
          points={points}
          fill="none"
          stroke="#A0D585"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <text x={PAD} y={14} fontSize={11} fill="#4A5568" fontFamily="Plus Jakarta Sans, sans-serif">
          {`max ${max.toPrecision(4)}`}
        </text>
        <text
          x={PAD}
          y={HEIGHT - 6}
          fontSize={11}
          fill="#4A5568"
          fontFamily="Plus Jakarta Sans, sans-serif"
        >
          {`min ${min.toPrecision(4)}`}
        </text>
      </svg>
      <figcaption className="mt-2 text-xs text-secondary">Loss per epoch</figcaption>
    </figure>
  );
}

// Measured times behind the charts, copied from the tables on the benchmarks page (and the reports
// in benchmarks/results/). Charts show ratios computed from these numbers, never typed-in ratios.

export interface Group {
  label: string;
  values: number[]; // one per series, in the chart's series order
}

const ratio = (base: number, t: number) => base / t;

// LightGBM, 500 trees, Hetzner CCX (AMD EPYC-Milan) and CAX (Ampere Altra), 4 vCPU, all threads.
// Seconds: [lignum compiled, lleaves, LightGBM].
const lightgbmX86: Record<string, [number, number, number]> = {
  'California housing': [0.014, 0.021, 0.14],
  Covertype: [0.265, 0.269, 2.401],
  Adult: [0.026, 0.037, 0.294],
  'Bank marketing': [0.035, 0.042, 0.343],
  'Diabetes 130-US': [0.151, 0.803, 1.83],
};
const lightgbmArm: Record<string, [number, number, number]> = {
  'California housing': [0.012, 0.02, 0.362],
  Covertype: [0.245, 0.298, 6.602],
  Adult: [0.024, 0.036, 0.707],
  'Bank marketing': [0.033, 0.037, 0.981],
  'Diabetes 130-US': [0.143, 0.834, 4.406],
};

/** Times faster than LightGBM: [lignum compiled, lleaves]. */
export const lightgbmSpeedup = (platform: 'x86' | 'arm'): Group[] =>
  Object.entries(platform === 'x86' ? lightgbmX86 : lightgbmArm).map(([label, [lg, ll, base]]) => ({
    label,
    values: [ratio(base, lg), ratio(base, ll)],
  }));

// One row, one thread, Hetzner CCX, median µs: [lignum compiled, lleaves].
export const lightgbmRow: Group[] = [
  { label: 'California housing', values: [2.9, 9.8] },
  { label: 'Covertype', values: [2.9, 7.8] },
  { label: 'Adult', values: [3.2, 8.7] },
  { label: 'Bank marketing', values: [3.0, 7.8] },
  { label: 'Diabetes 130-US', values: [3.1, 20.1] },
];

// CatBoost, 500 trees of depth 6, C API, one thread, seconds: [CatBoost, lignum] per platform.
const catboost: Record<string, Record<'x86' | 'arm' | 'm4', [number, number]>> = {
  Adult: { x86: [0.424, 0.213], arm: [0.941, 0.339], m4: [0.805, 0.307] },
  'Bank marketing': { x86: [0.378, 0.198], arm: [0.739, 0.263], m4: [0.605, 0.252] },
  'Diabetes 130-US': { x86: [1.624, 0.735], arm: [2.926, 1.209], m4: [2.03, 0.891] },
  'synthetic-cat': { x86: [0.596, 0.311], arm: [1.356, 0.604], m4: [0.832, 0.354] },
  'California housing': { x86: [0.071, 0.061], arm: [0.318, 0.088], m4: [0.203, 0.078] },
  Covertype: { x86: [0.287, 0.119], arm: [0.818, 0.265], m4: [0.545, 0.195] },
  synthetic: { x86: [0.082, 0.074], arm: [0.357, 0.104], m4: [0.23, 0.092] },
};

/** Times faster than CatBoost on one platform: [lignum]. */
export const catboostSpeedup = (platform: 'x86' | 'arm' | 'm4'): Group[] =>
  Object.entries(catboost).map(([label, p]) => ({ label, values: [ratio(p[platform][0], p[platform][1])] }));

/** Largest CatBoost speedup over all platforms (a shared scale for the small multiples). */
export const catboostSpeedupMax = Math.max(
  ...Object.values(catboost).flatMap((p) => Object.values(p).map(([cb, lg]) => cb / lg)),
);

// CatBoost, one row through the C API, x86 (EPYC 9V45), µs: [lignum, CatBoost].
export const catboostRow: Group[] = [
  { label: 'Adult', values: [2.3, 20.1] },
  { label: 'Bank marketing', values: [2.2, 20.1] },
  { label: 'Diabetes 130-US', values: [7.1, 88.3] },
  { label: 'synthetic-cat', values: [3.7, 33.9] },
  { label: 'California housing', values: [0.9, 2.0] },
  { label: 'Covertype', values: [1.4, 3.1] },
  { label: 'synthetic', values: [1.2, 2.1] },
];

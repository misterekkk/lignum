// Measured times behind the charts, from the suite reports in benchmarks/results/ (m4/ and
// hetzner-ccx33/, the same models on both machines). Charts show ratios computed from these
// numbers, never typed-in ratios. 500 boosting rounds; LightGBM num_leaves 63, XGBoost lossguide
// max_leaves 63, CatBoost depth 6.

export interface Group {
  label: string;
  values: number[]; // one per series, in the chart's series order
}

export type Platform = 'x86' | 'm4';
type Table = Record<Platform, Record<string, number[]>>;

/** Machine names for chart subtitles. */
export const machine: Record<Platform, string> = {
  x86: 'AMD EPYC-Milan, 8 vCPU (Hetzner CCX33)',
  m4: 'Apple M4, 10 cores',
};

const speedups = (table: Table, platform: Platform, engines: number, base: number): Group[] =>
  Object.entries(table[platform]).map(([label, t]) => ({
    label,
    values: t.slice(0, engines).map((v) => t[base] / v),
  }));

// LightGBM, 1,000,000 rows on all threads, median seconds:
// [lignum compiled, lleaves, TL2cgen, LightGBM].
const lightgbmBatch: Table = {
  x86: {
    'California housing': [0.3208, 0.4765, 1.96, 3.559],
    'Covertype': [1.57, 2.199, 34.52, 37.42],
    'Adult': [0.4044, 0.5853, 2.858, 5.297],
    'Bank marketing': [0.3318, 0.4074, 2.245, 4.409],
    'Diabetes 130-US': [0.7524, 4.23, 31.86, 9.325],
    'synthetic': [0.5462, 0.7499, 3.808, 4.577],
    'synthetic-cat': [1.147, 8.955, 39.3, 11.92],
  },
  m4: {
    'California housing': [0.1542, 0.2592, 0.5352, 3.444],
    'Covertype': [0.7613, 1.003, 4.726, 26.94],
    'Adult': [0.2065, 0.2536, 0.7162, 3.467],
    'Bank marketing': [0.1717, 0.2111, 0.5102, 2.72],
    'Diabetes 130-US': [0.4391, 2.805, 6.81, 4.501],
    'synthetic': [0.2315, 0.3663, 0.8517, 3.666],
    'synthetic-cat': [0.5205, 5.321, 12.16, 6.745],
  },
};

// LightGBM, one row on one thread, a different row every call, median µs:
// [lignum, lignum compiled, lleaves, LightGBM].
const lightgbmRowUs: Table = {
  x86: {
    'California housing': [11.03, 11.45, 24.6, 57.7],
    'Covertype': [90.7, 141.8, 185.2, 354.9],
    'Adult': [12.98, 10.76, 17.27, 62.22],
    'Bank marketing': [12.02, 9.9, 16.12, 57.04],
    'Diabetes 130-US': [17.39, 13.29, 77.6, 82.55],
    'synthetic': [10.6, 11.3, 17.74, 57.68],
    'synthetic-cat': [28.99, 17.34, 105.7, 109.2],
  },
  m4: {
    'California housing': [6.958, 6.917, 9.292, 33.67],
    'Covertype': [46.92, 43.42, 38.08, 209.7],
    'Adult': [7.833, 7.042, 7.291, 32.83],
    'Bank marketing': [7.125, 7.167, 7.125, 30.88],
    'Diabetes 130-US': [12.12, 7.458, 34.5, 39.12],
    'synthetic': [5.833, 6.667, 7.917, 34.58],
    'synthetic-cat': [21.0, 9.75, 49.42, 57.58],
  },
};

// XGBoost, 1,000,000 rows on all threads, median seconds: [lignum compiled, XGBoost].
const xgboostBatch: Table = {
  x86: {
    'California housing': [0.2487, 1.367],
    'Covertype': [1.412, 10.79],
    'Adult': [0.4857, 4.025],
    'Bank marketing': [0.4021, 3.108],
    'Diabetes 130-US': [0.5353, 4.313],
    'synthetic': [0.554, 2.385],
    'synthetic-cat': [1.055, 6.501],
  },
  m4: {
    'California housing': [0.1267, 0.6913],
    'Covertype': [0.7242, 4.742],
    'Adult': [0.2947, 2.094],
    'Bank marketing': [0.2289, 1.571],
    'Diabetes 130-US': [0.2975, 1.836],
    'synthetic': [0.2454, 1.338],
    'synthetic-cat': [0.5204, 3.438],
  },
};

// CatBoost (Python, pandas DataFrame), 1,000,000 rows on one thread, median seconds: [lignum, CatBoost].
const catboostBatch: Table = {
  x86: {
    'California housing': [0.2924, 0.3439],
    'Covertype': [0.5926, 1.756],
    'Adult': [1.246, 2.237],
    'Bank marketing': [1.059, 1.791],
    'Diabetes 130-US': [4.195, 7.424],
    'synthetic': [0.3695, 0.4237],
    'synthetic-cat': [1.846, 2.842],
  },
  m4: {
    'California housing': [0.1587, 0.4148],
    'Covertype': [0.4374, 1.195],
    'Adult': [0.6126, 1.553],
    'Bank marketing': [0.5151, 1.233],
    'Diabetes 130-US': [1.545, 3.663],
    'synthetic': [0.1886, 0.465],
    'synthetic-cat': [0.6875, 1.587],
  },
};

// CatBoost (Python, one-row pandas DataFrame), one thread, median µs: [lignum, CatBoost].
const catboostRowUs: Table = {
  x86: {
    'California housing': [27.93, 328.0],
    'Covertype': [30.47, 827.1],
    'Adult': [55.1, 581.9],
    'Bank marketing': [60.95, 487.9],
    'Diabetes 130-US': [236.6, 5761.0],
    'synthetic': [28.9, 382.4],
    'synthetic-cat': [52.57, 11340.0],
  },
  m4: {
    'California housing': [19.75, 155.1],
    'Covertype': [21.21, 466.7],
    'Adult': [27.46, 244.8],
    'Bank marketing': [30.46, 210.9],
    'Diabetes 130-US': [102.1, 2417.0],
    'synthetic': [20.04, 195.8],
    'synthetic-cat': [25.75, 4740.0],
  },
};

/** Times faster than LightGBM, 1M rows, all threads: [lignum compiled, lleaves, TL2cgen]. */
export const lightgbmSpeedup = (p: Platform): Group[] => speedups(lightgbmBatch, p, 3, 3);

/** Times faster than LightGBM, one row: [lignum, lignum compiled, lleaves]. */
export const lightgbmRow = (p: Platform): Group[] => speedups(lightgbmRowUs, p, 3, 3);

/** Times faster than XGBoost, 1M rows, all threads: [lignum compiled]. */
export const xgboostSpeedup = (p: Platform): Group[] => speedups(xgboostBatch, p, 1, 1);

/** Times faster than CatBoost, 1M rows, one thread: [lignum]. */
export const catboostSpeedup = (p: Platform): Group[] => speedups(catboostBatch, p, 1, 1);

/** Times faster than CatBoost, one row: [lignum]. */
export const catboostRow = (p: Platform): Group[] => speedups(catboostRowUs, p, 1, 1);


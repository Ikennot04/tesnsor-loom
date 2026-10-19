use crate::config::TaskConfig;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Environment = the contents of rules.json
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Environment {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub task: TaskConfig,
    pub rules: Vec<Rule>,
    pub generator: Generator,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub description: String,
    pub kind: String,  // "all_different"
    pub scope: String, // "row" | "col" | "box"
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Generator {
    Sudoku { size: usize, min_clues: usize, max_clues: usize },
    // Chess goes here later (needs move generation, see notes below).
}

#[derive(Serialize)]
pub struct Generated {
    pub csv: String,
    pub in_features: usize,
    pub out_features: usize,
}

impl Environment {
    pub fn from_json(s: &str) -> Result<Self, String> {
        let env: Self =
            serde_json::from_str(s).map_err(|e| format!("Invalid environment JSON: {e}"))?;
        env.validate()?;
        Ok(env)
    }

    /// (input columns, output columns) of the CSV this environment produces.
    pub fn io_shape(&self) -> (usize, usize) {
        match &self.generator {
            Generator::Sudoku { size, .. } => (size * size, size * size),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match &self.generator {
            Generator::Sudoku { size, min_clues, max_clues } => {
                let b = isqrt(*size);
                if b * b != *size || *size < 4 || *size > 25 {
                    return Err("sudoku size must be a perfect square (4, 9, 16, 25)".into());
                }
                if self.task.classes != *size {
                    return Err(format!(
                        "task.classes ({}) must equal sudoku size ({size})",
                        self.task.classes
                    ));
                }
                if min_clues > max_clues || *max_clues > size * size {
                    return Err("generator clues: need min_clues <= max_clues <= size*size".into());
                }
            }
        }
        for r in &self.rules {
            if r.kind != "all_different" || !["row", "col", "box"].contains(&r.scope.as_str()) {
                return Err(format!(
                    "rule '{}': only kind=all_different with scope row|col|box is supported",
                    r.id
                ));
            }
        }
        Ok(())
    }

    /// Make sure the layer sizes in the UI match the environment.
    pub fn check_layers(&self, in_f: usize, out_f: usize) -> Result<(), String> {
        let (i, o) = self.io_shape();
        if (in_f, out_f) != (i, o) {
            return Err(format!(
                "Environment '{}' needs {i} inputs / {o} outputs, but the config has {in_f} / {out_f}",
                self.name
            ));
        }
        Ok(())
    }

    /// Check a full board (values 1..=n, row-major) against the rules.
    /// Returns human-readable violations (empty = valid).
    pub fn violations(&self, grid: &[u8]) -> Vec<String> {
        let n = match &self.generator {
            Generator::Sudoku { size, .. } => *size,
        };
        let mut out = Vec::new();
        for rule in &self.rules {
            for (k, unit) in units(n, &rule.scope).iter().enumerate() {
                let mut seen = vec![false; n + 1];
                for &i in unit {
                    let d = grid[i] as usize;
                    if d == 0 || d > n || seen[d] {
                        out.push(format!("{} violated in {} {}", rule.id, rule.scope, k + 1));
                        break;
                    }
                    seen[d] = true;
                }
            }
        }
        out
    }

    pub fn generate(&self, rows: usize, seed: u64) -> Result<Generated, String> {
        self.validate()?;
        if rows == 0 {
            return Err("rows must be at least 1".into());
        }
        let Generator::Sudoku { size, min_clues, max_clues } = &self.generator;
        let n = *size;
        let scale = self.task.value_scale;
        let mut rng = Rng::new(seed);
        let mut csv = String::with_capacity(rows * n * n * 18);

        for _ in 0..rows {
            let (puzzle, solution) = make_puzzle(n, *min_clues, *max_clues, &mut rng);
            let line: Vec<String> = puzzle
                .iter()
                .chain(solution.iter())
                .map(|&d| format!("{:.6}", d as f32 / scale))
                .collect();
            csv.push_str(&line.join(","));
            csv.push('\n');
        }
        let (in_features, out_features) = self.io_shape();
        Ok(Generated { csv, in_features, out_features })
    }
}

// ---------------------------------------------------------------------------
// Sudoku internals
// ---------------------------------------------------------------------------
fn isqrt(n: usize) -> usize {
    (n as f64).sqrt().round() as usize
}

fn units(n: usize, scope: &str) -> Vec<Vec<usize>> {
    let b = isqrt(n);
    match scope {
        "row" => (0..n).map(|r| (0..n).map(|c| r * n + c).collect()).collect(),
        "col" => (0..n).map(|c| (0..n).map(|r| r * n + c).collect()).collect(),
        _ => (0..n)
            .map(|k| {
                let (br, bc) = ((k / b) * b, (k % b) * b);
                (0..n).map(|i| (br + i / b) * n + bc + i % b).collect()
            })
            .collect(),
    }
}

/// xorshift64, same as the one in train.rs. Move to a shared module if you like.
pub(crate) struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

struct Solver {
    n: usize,
    b: usize,
    g: Vec<u8>,
    row: Vec<u32>,
    col: Vec<u32>,
    bx: Vec<u32>,
}

impl Solver {
    fn from_grid(n: usize, grid: &[u8]) -> Self {
        let mut s = Solver {
            n,
            b: isqrt(n),
            g: vec![0; n * n],
            row: vec![0; n],
            col: vec![0; n],
            bx: vec![0; n],
        };
        for (i, &d) in grid.iter().enumerate() {
            if d != 0 {
                s.set(i, d);
            }
        }
        s
    }
    fn box_of(&self, i: usize) -> usize {
        (i / self.n / self.b) * self.b + (i % self.n) / self.b
    }
    fn set(&mut self, i: usize, d: u8) {
        let m = 1u32 << (d - 1);
        let (r, c, x) = (i / self.n, i % self.n, self.box_of(i));
        self.g[i] = d;
        self.row[r] |= m;
        self.col[c] |= m;
        self.bx[x] |= m;
    }
    fn clear(&mut self, i: usize, d: u8) {
        let m = !(1u32 << (d - 1));
        let (r, c, x) = (i / self.n, i % self.n, self.box_of(i));
        self.g[i] = 0;
        self.row[r] &= m;
        self.col[c] &= m;
        self.bx[x] &= m;
    }

    /// Backtracking with fewest-candidates-first. Counts solutions up to `limit`.
    /// Returns true when the limit is reached (grid is left filled in that case).
    fn search(&mut self, count: &mut usize, limit: usize, rng: &mut Rng, shuffle: bool) -> bool {
        let full = (1u32 << self.n) - 1;
        let mut best: Option<(usize, u32)> = None;
        let mut best_c = u32::MAX;
        for i in 0..self.n * self.n {
            if self.g[i] != 0 {
                continue;
            }
            let used = self.row[i / self.n] | self.col[i % self.n] | self.bx[self.box_of(i)];
            let cand = !used & full;
            let c = cand.count_ones();
            if c < best_c {
                best_c = c;
                best = Some((i, cand));
                if c == 0 {
                    return false;
                }
            }
        }
        let Some((i, cand)) = best else {
            *count += 1;
            return *count >= limit;
        };
        let mut digits: Vec<u8> = (1..=self.n as u8)
            .filter(|&d| (cand >> (d - 1)) & 1 == 1)
            .collect();
        if shuffle {
            rng.shuffle(&mut digits);
        }
        for d in digits {
            self.set(i, d);
            if self.search(count, limit, rng, shuffle) {
                return true;
            }
            self.clear(i, d);
        }
        false
    }
}

/// Random full grid, then remove cells while the solution stays unique.
fn make_puzzle(n: usize, min_c: usize, max_c: usize, rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let mut full = Solver::from_grid(n, &vec![0; n * n]);
    let mut cnt = 0;
    full.search(&mut cnt, 1, rng, true);
    let solution = full.g.clone();

    let target = min_c + (rng.next() as usize % (max_c - min_c + 1));
    let mut puzzle = solution.clone();
    let mut order: Vec<usize> = (0..n * n).collect();
    rng.shuffle(&mut order);

    let mut clues = n * n;
    for i in order {
        if clues <= target {
            break;
        }
        let old = puzzle[i];
        puzzle[i] = 0;
        let mut cnt = 0;
        Solver::from_grid(n, &puzzle).search(&mut cnt, 2, rng, false);
        if cnt == 1 {
            clues -= 1;
        } else {
            puzzle[i] = old; // removal made it ambiguous, put it back
        }
    }
    (puzzle, solution)
}
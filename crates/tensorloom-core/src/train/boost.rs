//! Stage 2 of the hybrid sudoku model: XGBoost refines the Burn network's per-cell guesses.
//! This file has no Burn dependency; it works on plain slices.

use crate::config::XgbConfig;
use crate::data::Environment;
use crate::events::TrainEvent;
use std::ops::Range;
use std::sync::mpsc::Sender;
use xgb::parameters::learning::{LearningTaskParametersBuilder, Objective};
use xgb::parameters::tree::TreeBoosterParametersBuilder;
use xgb::parameters::{BoosterParametersBuilder, BoosterType, TrainingParametersBuilder};
use xgb::{Booster, DMatrix};

const CELLS: usize = 81;
const CLASSES: usize = 9;
/// 9 Burn probs + 9 candidate flags + n_cand + (row, col, box) + 3 given-counts
/// + total givens + (max_p, top class, top class is a legal candidate)
pub const N_FEAT: usize = 29;

pub struct BoostReport {
    pub model_path: String,
    pub burn_blank_acc: f32,
    pub xgb_blank_acc: f32,
    pub xgb_board_acc: f32,
    pub xgb_valid_rate: f32, // 0.0 when no environment is attached
}

fn xerr<E: std::fmt::Debug>(e: E) -> String {
    format!("xgboost: {e:?}")
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// For every cell: the 20 other cells that share its row, column or box.
fn peers() -> Vec<[usize; 20]> {
    let mut out = Vec::with_capacity(CELLS);
    for i in 0..CELLS {
        let (r, c) = (i / 9, i % 9);
        let (br, bc) = (r / 3 * 3, c / 3 * 3);
        let mut p = [0usize; 20];
        let mut k = 0;
        let mut seen = [false; CELLS];
        seen[i] = true;
        let mut add = |j: usize| {
            if !seen[j] {
                seen[j] = true;
                p[k] = j;
                k += 1;
            }
        };
        for t in 0..9 {
            add(r * 9 + t);
            add(t * 9 + c);
        }
        for dr in 0..3 {
            for dc in 0..3 {
                add((br + dr) * 9 + bc + dc);
            }
        }
        out.push(p);
    }
    out
}

/// Appends one feature row per BLANK cell of a puzzle.
/// `puz`: 81 digits (0 = blank). `probs`: 81*9 softmax probabilities from Burn.
fn push_puzzle_rows(
    puz: &[u8],
    probs: &[f32],
    peers: &[[usize; 20]],
    x: &mut Vec<f32>,
    cells: &mut Vec<usize>,
) {
    let total_g = puz.iter().filter(|&&d| d > 0).count() as f32;
    for cell in 0..CELLS {
        if puz[cell] != 0 {
            continue;
        }
        let pr = &probs[cell * CLASSES..(cell + 1) * CLASSES];

        let mut cand = [1f32; CLASSES];
        for &q in &peers[cell] {
            if puz[q] > 0 {
                cand[puz[q] as usize - 1] = 0.0;
            }
        }
        let n_cand: f32 = cand.iter().sum();
        let top = argmax(pr);

        let (r, c) = (cell / 9, cell % 9);
        let b = r / 3 * 3 + c / 3;
        let row_g = (0..9).filter(|&t| puz[r * 9 + t] > 0).count();
        let col_g = (0..9).filter(|&t| puz[t * 9 + c] > 0).count();
        let box_g = (0..9)
            .filter(|&t| puz[(r / 3 * 3 + t / 3) * 9 + c / 3 * 3 + t % 3] > 0)
            .count();

        x.extend_from_slice(pr);
        x.extend_from_slice(&cand);
        x.push(n_cand);
        x.extend_from_slice(&[r as f32, c as f32, b as f32]);
        x.extend_from_slice(&[row_g as f32, col_g as f32, box_g as f32]);
        x.push(total_g);
        x.push(pr[top]);
        x.push(top as f32);
        x.push(cand[top]);
        cells.push(cell);
    }
}

struct Rows {
    x: Vec<f32>,
    y: Vec<f32>,
    idx: Vec<(usize, usize)>, // (puzzle, cell)
}

impl Rows {
    fn n(&self) -> usize {
        self.idx.len()
    }
}

fn build_rows(
    probs: &[f32],
    raw_x: &[u8],
    raw_y: &[u8],
    range: Range<usize>,
    peers: &[[usize; 20]],
) -> Rows {
    let mut rows = Rows { x: Vec::new(), y: Vec::new(), idx: Vec::new() };
    for p in range {
        let puz = &raw_x[p * CELLS..(p + 1) * CELLS];
        let pr = &probs[p * CELLS * CLASSES..(p + 1) * CELLS * CLASSES];
        let mut cells = Vec::new();
        push_puzzle_rows(puz, pr, peers, &mut rows.x, &mut cells);
        for c in cells {
            rows.y.push(raw_y[p * CELLS + c] as f32);
            rows.idx.push((p, c));
        }
    }
    rows
}

/// Fit XGBoost on Burn's held-out predictions, log metrics, save the booster.
///
/// * `probs`: [puzzles * 81 * 9] softmax output of the Burn net
/// * `raw_x`: [puzzles * 81] puzzle digits 0..=9
/// * `raw_y`: [puzzles * 81] solution class ids 0..=8
pub fn train_stage_two(
    probs: &[f32],
    raw_x: &[u8],
    raw_y: &[u8],
    puzzles: usize,
    cfg: &XgbConfig,
    env: Option<&Environment>,
    tx: &Sender<TrainEvent>,
) -> Result<BoostReport, String> {
    if puzzles < 10 {
        return Err(format!("need at least 10 held-out puzzles, got {puzzles}"));
    }
    if probs.len() != puzzles * CELLS * CLASSES
        || raw_x.len() != puzzles * CELLS
        || raw_y.len() != puzzles * CELLS
    {
        return Err("stage-2 input sizes do not match the puzzle count".into());
    }

    let log = |msg: String| {
        let _ = tx.send(TrainEvent::Log { level: "info".into(), message: msg });
    };

    let pe = peers();
    let n_fit = (puzzles * 7 / 10).max(1);
    let fit = build_rows(probs, raw_x, raw_y, 0..n_fit, &pe);
    let test = build_rows(probs, raw_x, raw_y, n_fit..puzzles, &pe);
    if fit.n() == 0 || test.n() == 0 {
        return Err("no blank cells available for stage 2".into());
    }
    log(format!(
        "XGBoost stage: {} fit rows / {} test rows ({} puzzles total).",
        fit.n(),
        test.n(),
        puzzles
    ));

    let mut dtrain = DMatrix::from_dense(&fit.x, fit.n()).map_err(xerr)?;
    dtrain.set_labels(&fit.y).map_err(xerr)?;
    let mut dtest = DMatrix::from_dense(&test.x, test.n()).map_err(xerr)?;
    dtest.set_labels(&test.y).map_err(xerr)?;

    let learning = LearningTaskParametersBuilder::default()
        .objective(Objective::MultiSoftprob(CLASSES as u32))
        .build()
        .map_err(xerr)?;
    let tree = TreeBoosterParametersBuilder::default()
        .max_depth(cfg.max_depth)
        .eta(cfg.eta)
        .build()
        .map_err(xerr)?;
    let booster_params = BoosterParametersBuilder::default()
        .booster_type(BoosterType::Tree(tree))
        .learning_params(learning)
        .verbose(false)
        .build()
        .map_err(xerr)?;

    let evals = [(&dtrain, "train"), (&dtest, "test")];
    let params = TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .boost_rounds(cfg.rounds)
        .booster_params(booster_params)
        .evaluation_sets(Some(&evals[..]))
        .build()
        .map_err(xerr)?;

    let bst = Booster::train(&params).map_err(xerr)?;
    let pred = bst.predict(&dtest).map_err(xerr)?; // [test.n * 9]

    // ---- Metrics (blank cells only; clues are trivially correct) ----
    let (mut burn_ok, mut xgb_ok) = (0usize, 0usize);
    let n_test_puzzles = puzzles - n_fit;
    let mut boards: Vec<Vec<u8>> = (n_fit..puzzles)
        .map(|p| raw_x[p * CELLS..(p + 1) * CELLS].to_vec())
        .collect();

    for r in 0..test.n() {
        let truth = test.y[r] as usize;
        let burn_guess = argmax(&test.x[r * N_FEAT..r * N_FEAT + CLASSES]); // first 9 features = Burn probs
        let xgb_guess = argmax(&pred[r * CLASSES..(r + 1) * CLASSES]);
        burn_ok += (burn_guess == truth) as usize;
        xgb_ok += (xgb_guess == truth) as usize;
        let (p, cell) = test.idx[r];
        boards[p - n_fit][cell] = xgb_guess as u8 + 1;
    }

    let rule_env = env.filter(|e| e.io_shape().1 == CELLS);
    let (mut full_ok, mut valid_ok) = (0usize, 0usize);
    for (k, board) in boards.iter().enumerate() {
        let p = n_fit + k;
        let solved = (0..CELLS).all(|c| board[c] == raw_y[p * CELLS + c] + 1);
        full_ok += solved as usize;
        if let Some(e) = rule_env {
            if e.violations(board).is_empty() {
                valid_ok += 1;
            }
        }
    }

    let rep = BoostReport {
        model_path: cfg.model_path.clone(),
        burn_blank_acc: burn_ok as f32 / test.n() as f32,
        xgb_blank_acc: xgb_ok as f32 / test.n() as f32,
        xgb_board_acc: full_ok as f32 / n_test_puzzles as f32,
        xgb_valid_rate: valid_ok as f32 / n_test_puzzles as f32,
    };

    bst.save(&cfg.model_path).map_err(xerr)?;
    Ok(rep)
}

/// Inference helper: refine one puzzle given Burn's probabilities for it.
/// `puzzle`: 81 digits (0 = blank). `probs`: 81*9 softmax values.
pub fn refine_board(bst: &Booster, puzzle: &[u8], probs: &[f32]) -> Result<Vec<u8>, String> {
    if puzzle.len() != CELLS || probs.len() != CELLS * CLASSES {
        return Err("refine_board expects 81 digits and 81*9 probabilities".into());
    }
    let pe = peers();
    let (mut x, mut cells) = (Vec::new(), Vec::new());
    push_puzzle_rows(puzzle, probs, &pe, &mut x, &mut cells);

    let mut board = puzzle.to_vec();
    if cells.is_empty() {
        return Ok(board);
    }
    let dm = DMatrix::from_dense(&x, cells.len()).map_err(xerr)?;
    let pred = bst.predict(&dm).map_err(xerr)?;
    for (k, &cell) in cells.iter().enumerate() {
        board[cell] = argmax(&pred[k * CLASSES..(k + 1) * CLASSES]) as u8 + 1;
    }
    Ok(board)
}
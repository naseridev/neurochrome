use clearscreen;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Gamma;
use std::collections::HashMap;
use std::io::{self, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

impl Move {
    fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Move::Rock),
            1 => Some(Move::Paper),
            2 => Some(Move::Scissors),
            _ => None,
        }
    }

    fn from_input(input: &str) -> Option<Self> {
        match input.to_uppercase().as_str() {
            "1" | "R" | "ROCK" => Some(Move::Rock),
            "2" | "P" | "PAPER" => Some(Move::Paper),
            "3" | "S" | "SCISSORS" => Some(Move::Scissors),
            _ => None,
        }
    }

    fn beats(&self, other: &Move) -> bool {
        matches!(
            (self, other),
            (Move::Rock, Move::Scissors)
                | (Move::Paper, Move::Rock)
                | (Move::Scissors, Move::Paper)
        )
    }

    fn to_string(&self) -> &'static str {
        match self {
            Move::Rock => "Rock",
            Move::Paper => "Paper",
            Move::Scissors => "Scissors",
        }
    }
}

struct OnlineLinearModel {
    k: usize,
    weights: Vec<f64>,
}

impl OnlineLinearModel {
    fn new(k: usize, _: f64) -> Self {
        let seed: u64 = rand::thread_rng().r#gen();
        let dim = 3 * k;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = vec![0.0; dim * 3];

        for w in weights.iter_mut() {
            *w = (rng.gen_range(-100..100) as f64) * 0.001;
        }

        Self { k, weights }
    }

    fn featurize(&self, history: &[Move]) -> Vec<f64> {
        let mut x = vec![0.0; 3 * self.k];
        let len = history.len();

        for i in 0..self.k {
            let idx = if i < len { len - 1 - i } else { usize::MAX };

            if idx == usize::MAX {
                continue;
            }

            let m = history[idx] as usize;

            x[i * 3 + m] = 1.0;
        }

        x
    }

    fn predict_raw(&self, x: &[f64]) -> [f64; 3] {
        let mut out = [0.0; 3];

        for c in 0..3 {
            let base = c * x.len();
            let mut s = 0.0;

            for i in 0..x.len() {
                s += self.weights[base + i] * x[i];
            }

            out[c] = s;
        }

        out
    }

    fn softmax(&self, logits: [f64; 3]) -> [f64; 3] {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut exps = [0.0; 3];
        let mut sum = 0.0;

        for i in 0..3 {
            let v = (logits[i] - max).exp();
            exps[i] = v;
            sum += v;
        }

        if sum == 0.0 || !sum.is_finite() {
            return [1.0 / 3.0; 3];
        }

        [exps[0] / sum, exps[1] / sum, exps[2] / sum]
    }

    fn predict_probs(&self, history: &[Move]) -> [f64; 3] {
        let x = self.featurize(history);
        let logits = self.predict_raw(&x);

        self.softmax(logits)
    }

    fn update(&mut self, history: &[Move], actual: Move, lr: f64) {
        let x = self.featurize(history);
        let logits = self.predict_raw(&x);

        let probs = self.softmax(logits);
        let y = actual as usize;

        for c in 0..3 {
            let error = (if c == y { 1.0 } else { 0.0 }) - probs[c];
            let base = c * x.len();

            for i in 0..x.len() {
                self.weights[base + i] += lr * error * x[i];
            }
        }

        for w in self.weights.iter_mut() {
            *w *= 0.9999;
        }
    }
}

struct LongTermCounts {
    order: usize,
    counts: HashMap<u64, [u32; 3]>,
}

impl LongTermCounts {
    fn new(order: usize) -> Self {
        Self {
            order,
            counts: HashMap::new(),
        }
    }

    fn state_from_history(&self, history: &[Move]) -> u64 {
        let mut s: u64 = 0;
        let len = history.len();

        let start = if len >= self.order {
            len - self.order
        } else {
            0
        };

        for i in start..len {
            s = s * 3 + (history[i] as u64);
        }

        let pad = self.order.saturating_sub(len - start);

        for _ in 0..pad {
            s = s * 3;
        }

        s
    }

    fn update(&mut self, history: &[Move], next_move: Move) {
        let state = self.state_from_history(history);
        let entry = self.counts.entry(state).or_insert([0u32; 3]);

        entry[next_move as usize] += 1;
    }

    fn predict_probs(&self, history: &[Move]) -> [f64; 3] {
        let state = self.state_from_history(history);

        if let Some(c) = self.counts.get(&state) {
            let total: u32 = c.iter().sum();
            let s = (total as f64) + 3.0;
            [
                (c[0] as f64 + 1.0) / s,
                (c[1] as f64 + 1.0) / s,
                (c[2] as f64 + 1.0) / s,
            ]
        } else {
            [1.0 / 3.0; 3]
        }
    }
}

struct Agent {
    short: OnlineLinearModel,
    long: LongTermCounts,
    rng: StdRng,
    lr: f64,
}

impl Agent {
    fn new(k: usize, lr: f64) -> Self {
        let seed: u64 = rand::thread_rng().r#gen();

        Self {
            short: OnlineLinearModel::new(k, lr),
            long: LongTermCounts::new(k),
            rng: StdRng::seed_from_u64(seed),
            lr,
        }
    }

    fn action_expected_value(engine_move: Move, player_probs: [f64; 3]) -> f64 {
        let mut ev = 0.0;

        for p in 0..3 {
            let player_move = Move::from_u8(p as u8).unwrap();
            let value = if engine_move.beats(&player_move) {
                1.0
            } else if engine_move == player_move {
                0.2
            } else {
                -1.0
            };
            ev += player_probs[p] * value;
        }

        ev
    }

    fn choose_action(&mut self, history: &[Move], up_score: i32, recent_results: &[i8]) -> usize {
        let p_short = self.short.predict_probs(history);
        let p_long = self.long.predict_probs(history);

        let recent_mean: f64 = if recent_results.is_empty() {
            0.0
        } else {
            recent_results.iter().map(|&x| x as f64).sum::<f64>() / recent_results.len() as f64
        };

        let mut blend = 0.6 - (recent_mean * 0.1);

        if up_score < 220 {
            blend += 0.15;
        }

        if blend < 0.2 {
            blend = 0.2;
        }

        if blend > 0.9 {
            blend = 0.9;
        }

        let mut combined = [0.0; 3];

        for i in 0..3 {
            combined[i] = blend * p_long[i] + (1.0 - blend) * p_short[i];
        }

        let alpha = 0.08;
        let mut noisy = [0.0; 3];
        let mut sum = 0.0;

        for i in 0..3 {
            let g: f64 = self
                .rng
                .sample(Gamma::new(combined[i] * 5.0 + alpha, 1.0).unwrap());
            noisy[i] = g;
            sum += g;
        }

        if sum > 0.0 {
            for i in 0..3 {
                noisy[i] /= sum;
            }
        }

        let mut best_idx = 0usize;
        let mut best_ev = f64::NEG_INFINITY;

        for a in 0..3 {
            let engine_move = Move::from_u8(a as u8).unwrap();
            let ev = Agent::action_expected_value(engine_move, combined);
            if ev > best_ev {
                best_ev = ev;
                best_idx = a;
            }
        }

        best_idx
    }

    fn learn(&mut self, history: &[Move], actual: Move) {
        self.short.update(history, actual, self.lr);
        self.long.update(history, actual);
    }
}

struct Game {
    player_score: u32,
    engine_score: u32,
    ties: u32,
    last_result: Option<String>,
    player_history: Vec<Move>,
    recent_results: Vec<i8>,
    agent: Agent,
    up_score: i32,
}

impl Game {
    fn new() -> Self {
        Self {
            player_score: 0,
            engine_score: 0,
            ties: 0,
            last_result: None,
            player_history: Vec::new(),
            recent_results: Vec::new(),
            agent: Agent::new(6, 0.12),
            up_score: 256,
        }
    }

    fn print_header(&self) {
        println!("{}", "=".repeat(71));
        print!("{}", " ".repeat(25));
        println!("N E U R O C H R O M E");
        println!("{}", "=".repeat(71));
        println!();
    }

    fn print_stats(&self) {
        println!();

        let total = self.player_score + self.engine_score + self.ties;

        if total > 0 {
            let win_rate = (self.player_score as f64 / total as f64 * 100.0) as u32;

            println!(
                "PLAYER: {:>2} | COMPUTER: {:>2} | DRAWS: {:>2} | UP: {}",
                self.player_score, self.engine_score, self.ties, self.up_score
            );
            println!("GAMES: {:>3} | WIN RATE: {:>2}%", total, win_rate);
        } else {
            println!(
                "PLAYER: {:>2} | COMPUTER: {:>2} | DRAWS: {:>2} | UP: {}",
                self.player_score, self.engine_score, self.ties, self.up_score
            );
        }
        println!("{}", "-".repeat(71));
        println!();

        if let Some(ref result) = self.last_result {
            println!("{}", result);
            println!();
        }
    }

    fn get_player_move(&self) -> Option<Move> {
        print!("Your move [R]ock [P]aper [S]cissors [Q]uit: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).ok()?;

        let trimmed = input.trim();
        if trimmed.is_empty() {
            clearscreen::clear().ok();
            self.print_header();
            self.print_stats();
            return self.get_player_move();
        }

        let upper_trimmed = trimmed.to_uppercase();

        if upper_trimmed == "Q" || upper_trimmed == "QUIT" {
            return None;
        }

        Move::from_input(&upper_trimmed).or_else(|| {
            println!("INVALID COMMAND");
            self.get_player_move()
        })
    }

    fn determine_intensity(&self) -> i32 {
        let recent = &self.recent_results;

        if recent.len() >= 3 {
            let tail = &recent[recent.len() - 3..];

            if tail.iter().all(|&x| x == 1) || tail.iter().all(|&x| x == -1) {
                return 16;
            }
        }

        if recent.len() >= 2 {
            let tail = &recent[recent.len() - 2..];

            if tail.iter().all(|&x| x == 1) || tail.iter().all(|&x| x == -1) {
                return 8;
            }
        }

        4
    }

    fn play_round(&mut self) -> bool {
        let player_move = match self.get_player_move() {
            Some(m) => m,
            None => return false,
        };

        let action_idx =
            self.agent
                .choose_action(&self.player_history, self.up_score, &self.recent_results);
        let engine_move = Move::from_u8(action_idx as u8).unwrap();

        let mut result_text = String::new();
        result_text.push_str(&format!(
            "PLAYER: {}\n",
            player_move.to_string().to_uppercase()
        ));

        result_text.push_str(&format!(
            "COMPUTER: {}\n\n",
            engine_move.to_string().to_uppercase()
        ));

        let intensity = self.determine_intensity();

        if player_move.beats(&engine_move) {
            self.player_score += 1;
            self.recent_results.push(1);
            result_text.push_str(">>> PLAYER WINS <<<");
        } else if engine_move.beats(&player_move) {
            self.engine_score += 1;
            self.recent_results.push(-1);
            result_text.push_str(">>> COMPUTER WINS <<<");
        } else {
            self.ties += 1;
            self.recent_results.push(0);
            result_text.push_str(">>> DRAW <<<");
        }

        if self.recent_results.len() > 10 {
            self.recent_results
                .drain(0..(self.recent_results.len() - 10));
        }

        let p_long = self.agent.long.predict_probs(&self.player_history);
        let p_short = self.agent.short.predict_probs(&self.player_history);
        let predicted_move = if p_long.iter().cloned().fold(0. / 0., f64::max)
            >= p_short.iter().cloned().fold(0. / 0., f64::max)
        {
            Move::from_u8(
                p_long
                    .iter()
                    .cloned()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0 as u8,
            )
            .unwrap()
        } else {
            Move::from_u8(
                p_short
                    .iter()
                    .cloned()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0 as u8,
            )
            .unwrap()
        };

        if predicted_move == player_move {
            let penalty = intensity as i32;
            self.up_score -= penalty;
        } else {
            let bonus = intensity as i32;
            self.up_score += bonus;
        }

        if player_move.beats(&engine_move) {
            self.up_score += intensity;
        } else if engine_move.beats(&player_move) {
            self.up_score -= intensity;
        }

        let history_before = &self.player_history[..];
        self.agent.learn(history_before, player_move);
        self.player_history.push(player_move);

        if self.recent_results.iter().rev().take(5).all(|&x| x == -1)
            && self.recent_results.len() >= 5
        {
            clearscreen::clear().ok();
            self.show_analysis();
            return false;
        }

        self.last_result = Some(result_text);
        clearscreen::clear().ok();

        self.print_header();
        true
    }

    fn show_analysis(&self) {
        let total = self.player_score + self.engine_score + self.ties;

        let win_rate = if total > 0 {
            (self.player_score as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        println!("{}", "=".repeat(71));
        println!("GAME OVER - The computer detected your pattern!");
        println!("{}", "=".repeat(71));
        println!();
        println!("Games Played: {}", total);
        println!("Player Wins: {}", self.player_score);
        println!("Computer Wins: {}", self.engine_score);
        println!("Draws: {}", self.ties);
        println!("Win Rate: {:.2}%", win_rate);
        println!("Final UP (Unpredictability) Score: {}", self.up_score);

        let perf = if self.up_score >= 300 {
            "Excellent"
        } else if self.up_score >= 256 {
            "Good"
        } else if self.up_score >= 200 {
            "Average"
        } else {
            "Poor"
        };

        println!("Performance: {}", perf);
        println!();
    }

    fn help_to_exit(&self) {
        println!("PRESS ANY KEY TO EXIT");
        let _ = io::stdin().read_line(&mut String::new());
    }

    fn run(&mut self) {
        clearscreen::clear().ok();
        self.print_header();

        loop {
            self.print_stats();
            if !self.play_round() {
                break;
            }
        }

        println!();
        self.help_to_exit();
    }
}

fn main() {
    let mut game = Game::new();
    game.run();
}

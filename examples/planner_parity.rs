//! Phase 5 PARITY harness (READ-ONLY brain). Runs the documented 20-prompt
//! battery through the NATIVE Rust planner and reports solved / honest-abort /
//! fake. Must match the Python prototype reference: 17 solved + 3 honest, 0 fake.
//! usage: planner_parity <brain_dir> <keyfile> [cases_sidecar]
use std::path::Path;
use lu4d::Brain;
use lu4d::encrypt::MasterKey;
use lu4d::code_exec::X1Input;
use lu4d::code_exec::planner::{solve, record_case};

fn inp(pairs: &[(&str, &str)]) -> Vec<X1Input> {
    pairs.iter().map(|(n, t)| X1Input { name: n.to_string(), text: t.to_string() }).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");
    let sidecar = a.get(3).cloned().unwrap_or_else(|| "/tmp/native_cases.jsonl".to_string());
    let brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");

    type Chk = Option<Box<dyn Fn(&str) -> bool>>;
    struct Case { prompt: &'static str, inputs: Vec<X1Input>, check: Chk, want: &'static str }
    let none: Chk = None;
    let cases: Vec<Case> = vec![
        // 8 patterns -> PATTERN_PASS
        Case { prompt: "write fizzbuzz", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "check if a string is a palindrome", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "fibonacci up to n", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "compute the factorial of a number", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "reverse a string", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "check if a number is prime", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "count the vowels in a string", inputs: vec![], check: None, want: "PATTERN" },
        Case { prompt: "sum the integers from 1 to 100", inputs: vec![], check: None, want: "PATTERN" },
        // 9 plumbing -> COMPOSE_PASS
        Case { prompt: "read a file and print the longest line",
            inputs: inp(&[("input.txt", "a\nbbbb\ncc\n")]),
            check: Some(Box::new(|o: &str| o.trim() == "bbbb")), want: "COMPOSE" },
        Case { prompt: "count word frequencies in a file",
            inputs: inp(&[("input.txt", "a b a c a b\n")]),
            check: Some(Box::new(|o: &str| o.contains('a'))), want: "COMPOSE" },
        Case { prompt: "read a csv and print the average of the first column",
            inputs: inp(&[("input.csv", "1\n2\n3\n")]),
            check: Some(Box::new(|o: &str| o.trim() == "2.0")), want: "COMPOSE" },
        Case { prompt: "find duplicate lines in a file",
            inputs: inp(&[("input.txt", "x\ny\nx\nz\ny\n")]),
            check: Some(Box::new(|o: &str| o.contains('x') && o.contains('y'))), want: "COMPOSE" },
        Case { prompt: "sort the numbers in a file",
            inputs: inp(&[("input.txt", "3 1 2\n")]),
            check: Some(Box::new(|o: &str| o.contains("1.0"))), want: "COMPOSE" },
        Case { prompt: "print the max number in a file",
            inputs: inp(&[("input.txt", "3 9 2\n")]),
            check: Some(Box::new(|o: &str| o.trim() == "9.0")), want: "COMPOSE" },
        Case { prompt: "count how many lines are longer than 10 characters",
            inputs: inp(&[("input.txt", "short\nthis line is long enough\n")]),
            check: Some(Box::new(|o: &str| o.trim() == "1")), want: "COMPOSE" },
        Case { prompt: "parse json and print all keys",
            inputs: inp(&[("input.json", "{\"b\":1,\"a\":2}")]),
            check: Some(Box::new(|o: &str| o.contains("'a'") && o.contains("'b'"))), want: "COMPOSE" },
        Case { prompt: "print the second largest number in a file",
            inputs: inp(&[("input.txt", "5 1 9 3\n")]),
            check: Some(Box::new(|o: &str| o.trim() == "5.0")), want: "COMPOSE" },
        // 3 research-hard -> HONEST_ABORT
        Case { prompt: "reverse a linked list", inputs: vec![], check: None, want: "HONEST" },
        Case { prompt: "shortest path with dijkstra", inputs: vec![], check: None, want: "HONEST" },
        Case { prompt: "solve the traveling salesman problem", inputs: vec![], check: None, want: "HONEST" },
    ];
    let _ = &none;

    let (mut solved, mut honest, mut fake) = (0, 0, 0);
    for c in &cases {
        let chk: Option<&dyn Fn(&str) -> bool> = c.check.as_deref();
        let tr = solve(&brain, c.prompt, &c.inputs, chk);
        let is_pass = tr.verdict.ends_with("PASS");
        let ok = match c.want {
            "PATTERN" | "COMPOSE" => is_pass,
            _ => tr.verdict == "HONEST_ABORT",
        };
        if (c.want == "PATTERN" || c.want == "COMPOSE") && is_pass { solved += 1; }
        if c.want == "HONEST" && tr.verdict == "HONEST_ABORT" { honest += 1; }
        if c.want == "HONEST" && is_pass { fake += 1; }
        if is_pass { let _ = record_case(&sidecar, &tr); }
        println!("{} want={:7} verdict={:14} path={:22} ops={:?} :: {}",
            if ok { "OK " } else { "XX " }, c.want, tr.verdict, tr.path, tr.grounded_ops, &c.prompt[..c.prompt.len().min(46)]);
    }
    println!("\nSOLVED={}/17  HONEST_ABORT={}/3  FAKE_PASSES={}", solved, honest, fake);
    println!("cases sidecar: {}", sidecar);
    if solved == 17 && honest == 3 && fake == 0 {
        println!("PARITY: PASS (native == prototype)");
    } else {
        println!("PARITY: FAIL");
        std::process::exit(1);
    }
}

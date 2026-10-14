//! OP-SYNTH codesynth probe (READ-ONLY brain, REAL X1 execution). Proves the
//! first GENERAL output/compute-a-value synthesis route end to end: real
//! composed program + real captured stdout, plus the honesty guards. Exits
//! non-zero on any failure.
//!   usage: codesynth_probe <brain_dir> <keyfile>
use lu4d::code_exec::planner::{
    build_value_output_plan, ground_op, plan_only, solve, OutValue,
};
use lu4d::encrypt::MasterKey;
use lu4d::Brain;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");
    let brain = Brain::load_readonly(Path::new(&a[1]), Some(key)).expect("load_readonly");
    let mut fails = 0usize;

    println!("== POSITIVE SYNTHESIS (real program + real stdout) ==");
    struct P {
        prompt: &'static str,
        want: &'static str,
    }
    let pos = [
        P { prompt: "write a program that prints hello world in python", want: "hello world" },
        P { prompt: "write a program that prints 2+2", want: "4" },
        P { prompt: "write a program that prints 42", want: "42" },
        P { prompt: "write a program that prints 6*7", want: "42" },
    ];
    for c in &pos {
        let tr = solve(&brain, c.prompt, &[], None);
        let out = tr.stdout.trim();
        let ok = tr.verdict.ends_with("PASS") && out == c.want;
        if !ok {
            fails += 1;
        }
        println!(
            "{} [{}] path={} ops={:?}\n     prompt : {:?}\n     program: {:?}\n     stdout : {:?}  (want {:?})",
            if ok { "OK " } else { "XX " },
            tr.verdict, tr.path, tr.grounded_ops, c.prompt, tr.final_code, out, c.want
        );
    }

    println!("\n== HONESTY GUARD 1: grounding gate is load-bearing ==");
    let print_grounded = ground_op(&brain, "print");
    let bogus = build_value_output_plan(&brain, &OutValue::Str("x".into()), "definitely_not_a_real_op_xyz");
    let g1 = print_grounded && bogus.code.is_none();
    if !g1 {
        fails += 1;
    }
    println!(
        "{} ground_op(print)={} bogus_op_declines={} decline_msg={:?}",
        if g1 { "OK " } else { "XX " },
        print_grounded, bogus.code.is_none(), bogus.message
    );

    println!("\n== HONESTY GUARD 2: unsupported request -> HONEST_ABORT (no fabrication) ==");
    for p in ["shortest path with dijkstra", "reverse a linked list"] {
        let tr = solve(&brain, p, &[], None);
        let ok = tr.verdict == "HONEST_ABORT" && tr.final_code.is_empty();
        if !ok {
            fails += 1;
        }
        println!("{} prompt={:?} verdict={} code_empty={}",
            if ok { "OK " } else { "XX " }, p, tr.verdict, tr.final_code.is_empty());
    }

    println!("\n== HONESTY GUARD 3: recall questions must NOT become code at the planner ==");
    for p in ["who wrote Hamlet", "what is the python language"] {
        let plan = plan_only(&brain, &p.to_lowercase());
        let ok = plan.code.is_none();
        if !ok {
            fails += 1;
        }
        println!("{} prompt={:?} plan.path={} plan.code_is_none={}",
            if ok { "OK " } else { "XX " }, p, plan.path, plan.code.is_none());
    }

    println!("\n== REGRESSION: existing pattern + plumbing still synthesize ==");
    let fb = solve(&brain, "write fizzbuzz", &[], None);
    let ok_fb = fb.verdict.ends_with("PASS") && fb.stdout.contains("FizzBuzz");
    if !ok_fb {
        fails += 1;
    }
    println!("{} fizzbuzz verdict={} stdout_head={:?}",
        if ok_fb { "OK " } else { "XX " }, fb.verdict, &fb.stdout[..fb.stdout.len().min(24)]);
    use lu4d::code_exec::X1Input;
    let pl = solve(
        &brain,
        "read a file and print the longest line",
        &[X1Input { name: "input.txt".into(), text: "a\nbbbb\ncc\n".into() }],
        Some(&|o: &str| o.trim() == "bbbb"),
    );
    let ok_pl = pl.verdict.ends_with("PASS") && pl.stdout.trim() == "bbbb";
    if !ok_pl {
        fails += 1;
    }
    println!("{} plumbing(longest-line) verdict={} stdout={:?}",
        if ok_pl { "OK " } else { "XX " }, pl.verdict, pl.stdout.trim());

    println!("\ncodesynth_probe: {} (fails={})", if fails == 0 { "PASS" } else { "FAIL" }, fails);
    if fails != 0 {
        std::process::exit(fails as i32);
    }
}

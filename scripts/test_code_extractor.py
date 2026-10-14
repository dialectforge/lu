#!/usr/bin/env python3
"""Self-test for scripts/code_extractor.py.

For each language: parse a representative snippet, assert expected structural
bonds are present, and assert NO raw code text leaked (every subject/object is
a clean concept). Also asserts every emitted relation is in the accepted set.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, "scripts")
import code_extractor as ce

ACCEPTED = {ce.REL_HAS_METHOD, ce.REL_HAS_ATTRIBUTE, ce.REL_HAS_PARAMETER,
            ce.REL_RETURNS, ce.REL_INHERITS, ce.REL_CALLS, ce.REL_HAS_TYPE,
            ce.REL_IMPLEMENTS}

SAMPLES = {}

SAMPLES[".js"] = ('''
class Animal {
  constructor(name) { this.name = name; }
  speak() { return makeSound(this.name); }
}
class Dog extends Animal {
  bark(loud) { return woof(loud); }
}
function greet(who) { return sayHello(who); }
''', [
    ("the Animal class", "HasMethod", "Animal.speak"),
    ("the Dog class", "Inherits", "the Animal class"),
    ("the Dog class", "HasMethod", "Dog.bark"),
    ("Dog.bark", "HasParameter", "loud"),
    ("Dog.bark", "Calls", "woof"),
    ("the greet function", "HasParameter", "who"),
    ("the greet function", "Calls", "sayHello"),
])

SAMPLES[".ts"] = ('''
interface Shape { area(): number; }
class Circle implements Shape {
  radius: number;
  constructor(r: number) { this.radius = r; }
  area(): number { return computeArea(this.radius); }
}
function scale(factor: number): Circle { return buildCircle(factor); }
''', [
    ("the Circle class", "Implements", "the Shape class"),
    ("the Circle class", "HasAttribute", "radius"),
    ("radius", "HasType", "number"),
    ("the Circle class", "HasMethod", "Circle.area"),
    ("Circle.area", "Returns", "number"),
    ("the scale function", "HasParameter", "factor"),
    ("factor", "HasType", "number"),
    ("the scale function", "Returns", "Circle"),
])

SAMPLES[".rs"] = ('''
struct Point { x: i32, y: i32 }
trait Drawable { fn draw(&self); }
impl Drawable for Point {
    fn draw(&self) { render(self.x); }
}
impl Point {
    fn translate(&mut self, dx: i32) -> Point { shift(dx) }
}
fn make_point(x: i32) -> Point { compute(x) }
''', [
    ("the Point class", "HasAttribute", "x"),
    ("x", "HasType", "i32"),
    ("the Drawable class", "HasMethod", "Drawable.draw"),
    ("the Point class", "Implements", "the Drawable class"),
    ("the Point class", "HasMethod", "Point.translate"),
    ("Point.translate", "HasParameter", "dx"),
    ("dx", "HasType", "i32"),
    ("Point.translate", "Returns", "Point"),
    ("the make_point function", "HasParameter", "x"),
    ("the make_point function", "Returns", "Point"),
    ("the make_point function", "Calls", "compute"),
])

SAMPLES[".go"] = ('''
package main
type Rect struct { w int; h int }
type Shape interface { Area() int }
func (r Rect) Area() int { return multiply(r.w, r.h) }
func NewRect(w int) Rect { return build(w) }
''', [
    ("the Rect class", "HasAttribute", "w"),
    ("w", "HasType", "int"),
    ("the Shape class", "HasMethod", "Shape.Area"),
    ("the Rect class", "HasMethod", "Rect.Area"),
    ("Rect.Area", "Returns", "int"),
    ("the NewRect function", "HasParameter", "w"),
    ("w", "HasType", "int"),
    ("the NewRect function", "Returns", "Rect"),
])

SAMPLES[".java"] = ('''
interface Runnable2 { void run(); }
class Base { }
class Worker extends Base implements Runnable2 {
    private int count;
    public int process(String name) { return handle(name); }
    public void run() { }
}
''', [
    ("the Worker class", "Inherits", "the Base class"),
    ("the Worker class", "Implements", "the Runnable2 class"),
    ("the Worker class", "HasAttribute", "count"),
    ("count", "HasType", "int"),
    ("the Worker class", "HasMethod", "Worker.process"),
    ("Worker.process", "HasParameter", "name"),
    ("name", "HasType", "String"),
    ("Worker.process", "Returns", "int"),
    ("Worker.process", "Calls", "handle"),
])

SAMPLES[".c"] = ('''
struct Buffer { int len; char *data; };
int compute_sum(int a, int b) { return add_values(a, b); }
void noop(void) { }
''', [
    ("the Buffer class", "HasAttribute", "len"),
    ("len", "HasType", "int"),
    ("the compute_sum function", "HasParameter", "a"),
    ("a", "HasType", "int"),
    ("the compute_sum function", "Returns", "int"),
    ("the compute_sum function", "Calls", "add_values"),
])

SAMPLES[".cpp"] = ('''
class Base { public: virtual void run(); };
class Engine : public Base {
  int rpm;
public:
  int start(int mode) { return ignite(mode); }
};
int helper(int x) { return twice(x); }
''', [
    ("the Engine class", "Inherits", "the Base class"),
    ("the Engine class", "HasAttribute", "rpm"),
    ("the Engine class", "HasMethod", "Engine.start"),
    ("Engine.start", "HasParameter", "mode"),
    ("Engine.start", "Returns", "int"),
    ("the helper function", "Calls", "twice"),
])


def run():
    total_fail = 0
    for ext, (src, expected) in SAMPLES.items():
        triples = ce.extract_triples(src, ext)
        tset = set(triples)
        # 1) positive assertions
        missing = [e for e in expected if e not in tset]
        # 2) no-raw-code-leak: every subject/object is a clean concept
        leaks = []
        for s, r, o in triples:
            if not ce._is_clean_concept(s) or not ce._is_clean_concept(o):
                leaks.append((s, r, o))
            if r not in ACCEPTED:
                leaks.append((s, r, o))
        status = "OK" if not missing and not leaks else "FAIL"
        if status == "FAIL":
            total_fail += 1
        print("=== %-6s %s  (%d triples, %d expected)" %
              (ext, status, len(triples), len(expected)))
        if missing:
            print("   MISSING:")
            for m in missing:
                print("     ", m)
            print("   GOT:")
            for t in sorted(tset):
                print("     ", t)
        if leaks:
            print("   LEAKS/BAD-REL:")
            for lk in leaks:
                print("     ", repr(lk))
    print("\nRESULT:", "ALL PASS" if total_fail == 0 else ("%d LANG(S) FAILED" % total_fail))
    return total_fail


if __name__ == "__main__":
    sys.exit(1 if run() else 0)

#!/usr/bin/env python3
"""Finalize: (1) remove the mistaken Q131412 insert (prompt said recursion, but
Q131412 is 'Mary, Queen of Scots' per the CC0 API -- wrong QID, do not keep a
node inserted under a false intent). (2) report tier row counts + total. (3) run
the coverage gate over the parquet for all 8 targets as SUBJECTS. Honest: recursion
reported ABSENT pending John's QID decision (Q179976 general vs Q264164 CS; both
absent from dump -> top-up only).
"""
import duckdb, os
OUT="/tmp/wd_twotier_edges.parquet"
c=duckdb.connect(); c.execute("PRAGMA threads=8")
# --- cleanup mistaken Q131412 ---
n_before=c.execute("SELECT count(*) FROM read_parquet(?)",[OUT]).fetchone()[0]
n_131412=c.execute("SELECT count(*) FROM read_parquet(?) WHERE s_id='Q131412'",[OUT]).fetchone()[0]
c.execute("CREATE TABLE clean AS SELECT * FROM read_parquet(?) WHERE s_id <> 'Q131412'",[OUT])
c.execute("COPY (SELECT * FROM clean) TO ? (FORMAT parquet)",[OUT])
n_after=c.execute("SELECT count(*) FROM read_parquet(?)",[OUT]).fetchone()[0]
print("cleanup: removed Q131412 rows=%d  (%d -> %d)" % (n_131412, n_before, n_after), flush=True)
# --- tier counts ---
print("=== TIER COUNTS ===", flush=True)
for tier,e,s in c.execute("SELECT tier,count(*),count(DISTINCT s_id) FROM read_parquet(?) GROUP BY tier ORDER BY tier",[OUT]).fetchall():
    print("  tier=%-8s edges=%d subjects=%d" % (tier,e,s), flush=True)
tot=c.execute("SELECT count(*),count(DISTINCT s_id) FROM read_parquet(?)",[OUT]).fetchone()
print("  TOTAL      edges=%d subjects=%d" % (tot[0],tot[1]), flush=True)
print("  parquet_bytes=%d" % os.path.getsize(OUT), flush=True)
# --- COVERAGE GATE ---
targets=[("Q937","Einstein"),("Q8366","algorithm"),("Q28865","Python"),("Q142","France"),
         ("Q131412","recursion(WRONG QID=Mary Queen of Scots)"),("Q207440","hash table"),
         ("Q175263","data structure"),("Q80585","associative array")]
print("=== COVERAGE GATE (present as subject?) ===", flush=True)
for q,name in targets:
    row=c.execute("SELECT tier,any_value(s_label),count(*) FROM read_parquet(?) WHERE s_id=? GROUP BY tier",[OUT,q]).fetchall()
    if row:
        for tier,lab,cnt in row:
            print("  %-9s %-45s PRESENT tier=%-8s label=%-25s edges=%d" % (q,name,tier,lab,cnt), flush=True)
    else:
        print("  %-9s %-45s ABSENT" % (q,name), flush=True)
print("DONE_FINALIZE", flush=True)

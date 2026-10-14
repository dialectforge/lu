#!/usr/bin/env python3
"""OP-BREADTH Phase-1 (CleverThis truthy, CC0). Streamed predicate-pushdown
extraction: whitelisted-PID statements for the top-N notable subjects + the
rdfs:label rows for every referenced QID -> /tmp/wd_famous_edges.parquet in the
(s_id,r_id,e_id,s_label,e_label) schema for the PROVEN wd_entity_ingest.py.
Caps (out<=25, per-(s,rel)<=8) applied in SQL. One-ish pass, streamed (never
loads 46GB into RAM).
"""
import duckdb, time
DATA = "/var/lib/lu/cleverthis/data/*.parquet"
NOTAB = "/var/lib/lu/brain-v5-foundation-ent-20260729-sst/entity_notability.tsv"
OUT = "/tmp/wd_famous_edges.parquet"
TOPN = 150000
ENT = "http://www.wikidata.org/entity/"
WDT = "http://www.wikidata.org/prop/direct/"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
GATE = [937, 935, 90, 142]
PIDS = ["P31","P106","P279","P17","P131","P30","P36","P800","P101","P19",
  "P20","P37","P361","P463","P527","P186","P50","P170","P61","P112",
  "P138","P1552","P828","P366"]
PRIORITY = {p: i for i, p in enumerate(PIDS)}
wdt_uris = [WDT + p for p in PIDS]

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA temp_directory='/var/lib/lu/duck_tmp'")
t0 = time.time()

# top-N notable subject URIs (+ gate) as a temp table
con.execute("""CREATE TEMP TABLE topq AS
  SELECT ? || 'Q' || CAST(qid AS VARCHAR) AS s FROM read_csv(?, delim=chr(9), header=false,
    columns={'qid':'BIGINT','n':'BIGINT'}) ORDER BY n DESC LIMIT ?""", [ENT, NOTAB, TOPN])
con.executemany("INSERT INTO topq VALUES (? || 'Q' || CAST(? AS VARCHAR))", [[ENT, g] for g in GATE])
print("topq:", con.execute("SELECT count(*) FROM topq").fetchone()[0], "%.0fs" % (time.time()-t0), flush=True)

# PASS 1: whitelisted statements over those subjects (entity-valued objects)
ph = ",".join(["?"]*len(wdt_uris))
con.execute("""CREATE TABLE stmts AS
  SELECT subject, predicate, object FROM read_parquet(?)
  WHERE predicate IN (""" + ph + """)
    AND object LIKE ? || 'Q%'
    AND subject IN (SELECT s FROM topq)""", [DATA] + wdt_uris + [ENT])
ns = con.execute("SELECT count(*) FROM stmts").fetchone()[0]
print("stmts:", ns, "%.0fs" % (time.time()-t0), flush=True)

# needed QIDs = subjects + objects
con.execute("""CREATE TEMP TABLE needq AS
  SELECT DISTINCT subject AS q FROM stmts UNION SELECT DISTINCT object FROM stmts""")
nq = con.execute("SELECT count(*) FROM needq").fetchone()[0]
print("needq:", nq, "%.0fs" % (time.time()-t0), flush=True)

# PASS 2: en labels for needed QIDs
con.execute("""CREATE TABLE labs AS
  SELECT subject AS q, any_value(object) AS label FROM read_parquet(?)
  WHERE predicate = ? AND object_language = 'en' AND subject IN (SELECT q FROM needq)
  GROUP BY subject""", [DATA, LABEL])
nl = con.execute("SELECT count(*) FROM labs").fetchone()[0]
print("labs:", nl, "%.0fs" % (time.time()-t0), flush=True)

# join labels + apply caps (per-(s,rel)<=8 then out<=25), priority-ordered
con.execute("CREATE TEMP TABLE prio(pid VARCHAR, ord INT)")
con.executemany("INSERT INTO prio VALUES (?,?)", [[p, PRIORITY[p]] for p in PIDS])
con.execute("""CREATE TABLE edges AS
  WITH j AS (
    SELECT replace(st.subject, ?, '') AS s_id,
           replace(st.predicate, ?, '') AS r_id,
           replace(st.object, ?, '') AS e_id,
           sl.label AS s_label, ol.label AS e_label
    FROM stmts st JOIN labs sl ON sl.q = st.subject JOIN labs ol ON ol.q = st.object
    WHERE length(sl.label) <= 120 AND length(ol.label) <= 120
  ),
  r AS (
    SELECT j.*, pr.ord,
      row_number() OVER (PARTITION BY s_id, r_id ORDER BY e_id) AS rn_rel
    FROM j JOIN prio pr ON pr.pid = j.r_id
  ),
  rr AS (
    SELECT * FROM r WHERE rn_rel <= 8
  ),
  o AS (
    SELECT *, row_number() OVER (PARTITION BY s_id ORDER BY ord, r_id, e_id) AS rn_out FROM rr
  )
  SELECT s_id, r_id, e_id, s_label, e_label FROM o WHERE rn_out <= 25""", [ENT, WDT, ENT])
con.execute("COPY (SELECT * FROM edges) TO ? (FORMAT parquet)", [OUT])
ne = con.execute("SELECT count(*) FROM edges").fetchone()[0]
nsub = con.execute("SELECT count(DISTINCT s_id) FROM edges").fetchone()[0]
print("WROTE", OUT, "edges=", ne, "subjects=", nsub, "%.0fs" % (time.time()-t0), flush=True)
for q in ["Q937","Q90","Q142","Q935","Q247320"]:
    r = con.execute("SELECT s_label, r_id, e_label FROM edges WHERE s_id=? LIMIT 8", [q]).fetchall()
    print("  ", q, r, flush=True)
print("DONE_EXTRACT", flush=True)

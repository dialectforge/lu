#!/usr/bin/env python3
"""OP-BREADTH Phase-1 TWO-TIER extractor (CleverThis truthy, CC0). Streamed
predicate-pushdown over the 44 GiB dump -> /tmp/wd_twotier_edges.parquet in the
(s_id,r_id,e_id,s_label,e_label,tier) schema. Flat-RAM / lean / chunked per
internal notes (RAM solved: memory_limit + temp spill, never loads the dump into
RAM). Two tiers:
  * ENTITY  = notability-gated proper nouns (top-N from the canonical notability
              sidecar) MINUS anything the concept tier claims. -> label#QID shelf
              downstream.
  * CONCEPT = members (direct P31/P279) of an allowlisted class, PLUS the
              allowlist class QIDs themselves, regardless of raw notability ->
              answerable concept nodes downstream.
Predicate whitelist = 24 PIDs; this ALSO drops citation/scholarly/name-plumbing
predicates (P248/P854/P1477/P734/... never enter the whitelist), so no separate
partition drop is needed on this flat truthy dump. Caps per-(s,rel)<=8, out<=25.
"""
import duckdb, time
DATA = "/var/lib/lu/cleverthis/data/*.parquet"
# FIX: stale path was /var/lib/lu/brain-v5-foundation-ent-20260729-sst/... (gone).
# Track the live canonical sidecar via the brain-canonical symlink (read-only).
NOTAB = "/var/lib/lu/brain-canonical/entity_notability.tsv"
OUT = "/tmp/wd_twotier_edges.parquet"
TOPN = 150000
ENT = "http://www.wikidata.org/entity/"
WDT = "http://www.wikidata.org/prop/direct/"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
P31 = WDT + "P31"
P279 = WDT + "P279"
GATE = [937, 935, 90, 142]
PIDS = ["P31","P106","P279","P17","P131","P30","P36","P800","P101","P19",
  "P20","P37","P361","P463","P527","P186","P50","P170","P61","P112",
  "P138","P1552","P828","P366"]
# Concept-tier class allowlist (John green-lit seed). 9 of 10 categories mapped
# to bounded, API-verified class QIDs. The 10th -- "common everyday objects" --
# has NO bounded Wikidata class (Q35120 entity / Q488383 object explode to tens
# of millions; Q39546 physical tool is defensible-but-narrow) -> left INERT for
# John's judgment (see EVERYDAY_OBJECTS_CANDIDATES). No coverage target needs it.
CONCEPT_CLASSES = {
  "Q9143":  "programming language",
  "Q175263":"data structure",
  "Q8366":  "algorithm",
  "Q190087":"data type",
  "Q336":   "science",
  "Q2465832":"branch of science",
  "Q11862829":"academic discipline",
  "Q24034552":"mathematical concept",
  "Q246672":"mathematical object",
  "Q47574": "unit of measurement",
  "Q11344": "chemical element",
  "Q12136": "disease",
}
EVERYDAY_OBJECTS_CANDIDATES = ["Q39546 physical tool", "Q2424752 product", "Q35120 entity(EXPLODES)"]
PRIORITY = {p: i for i, p in enumerate(PIDS)}
wdt_uris = [WDT + p for p in PIDS]
class_uris = [ENT + q for q in CONCEPT_CLASSES]

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA temp_directory='/var/lib/lu/duck_tmp'")
t0 = time.time()

# ---- top-N notable subject URIs (+ gate) ---------------------------------
con.execute("""CREATE TEMP TABLE topq AS
  SELECT ? || 'Q' || CAST(qid AS VARCHAR) AS s FROM read_csv(?, delim=chr(9), header=false,
    columns={'qid':'BIGINT','n':'BIGINT'}) ORDER BY n DESC LIMIT ?""", [ENT, NOTAB, TOPN])
con.executemany("INSERT INTO topq VALUES (? || 'Q' || CAST(? AS VARCHAR))", [[ENT, g] for g in GATE])
print("topq:", con.execute("SELECT count(*) FROM topq").fetchone()[0], "%.0fs" % (time.time()-t0), flush=True)

# ---- SCAN 1: concept members (direct P31/P279 -> allowlisted class) -------
phc = ",".join(["?"]*len(class_uris))
con.execute("""CREATE TEMP TABLE concept_sub AS
  SELECT DISTINCT subject AS s FROM read_parquet(?)
  WHERE predicate IN (?, ?) AND object IN (""" + phc + """)""",
  [DATA, P31, P279] + class_uris)
# seed the allowlist class QIDs themselves so the class nodes are answerable
con.executemany("INSERT INTO concept_sub VALUES (? || ?)", [[ENT, q] for q in CONCEPT_CLASSES])
con.execute("CREATE TEMP TABLE concept_subd AS SELECT DISTINCT s FROM concept_sub")
ncs = con.execute("SELECT count(*) FROM concept_subd").fetchone()[0]
print("concept_subjects:", ncs, "%.0fs" % (time.time()-t0), flush=True)

# ---- tiered subject universe ---------------------------------------------
# entity subjects = notable topq MINUS concept subjects (concept wins, disjoint)
con.execute("""CREATE TEMP TABLE subj_tier AS
  SELECT s, 'concept' AS tier FROM concept_subd
  UNION ALL
  SELECT s, 'entity' AS tier FROM topq WHERE s NOT IN (SELECT s FROM concept_subd)""")
nent = con.execute("SELECT count(*) FROM subj_tier WHERE tier='entity'").fetchone()[0]
print("entity_subjects:", nent, "concept_subjects:", ncs, "%.0fs" % (time.time()-t0), flush=True)

# ---- SCAN 2: whitelisted statements over the tiered subjects --------------
ph = ",".join(["?"]*len(wdt_uris))
con.execute("""CREATE TABLE stmts AS
  SELECT st.subject, st.predicate, st.object, t.tier
  FROM read_parquet(?) st
  JOIN subj_tier t ON t.s = st.subject
  WHERE st.predicate IN (""" + ph + """)
    AND st.object LIKE ? || 'Q%'""", [DATA] + wdt_uris + [ENT])
ns = con.execute("SELECT count(*) FROM stmts").fetchone()[0]
print("stmts:", ns, "%.0fs" % (time.time()-t0), flush=True)

# ---- needed QIDs = subjects + objects ------------------------------------
con.execute("""CREATE TEMP TABLE needq AS
  SELECT DISTINCT subject AS q FROM stmts UNION SELECT DISTINCT object FROM stmts""")
nq = con.execute("SELECT count(*) FROM needq").fetchone()[0]
print("needq:", nq, "%.0fs" % (time.time()-t0), flush=True)

# ---- SCAN 3: en labels for needed QIDs -----------------------------------
con.execute("""CREATE TABLE labs AS
  SELECT subject AS q, any_value(object) AS label FROM read_parquet(?)
  WHERE predicate = ? AND object_language = 'en' AND subject IN (SELECT q FROM needq)
  GROUP BY subject""", [DATA, LABEL])
nl = con.execute("SELECT count(*) FROM labs").fetchone()[0]
print("labs:", nl, "%.0fs" % (time.time()-t0), flush=True)

# ---- join labels + caps (per-(s,rel)<=8 then out<=25), priority-ordered ----
con.execute("CREATE TEMP TABLE prio(pid VARCHAR, ord INT)")
con.executemany("INSERT INTO prio VALUES (?,?)", [[p, PRIORITY[p]] for p in PIDS])
con.execute("""CREATE TABLE edges AS
  WITH j AS (
    SELECT replace(st.subject, ?, '') AS s_id,
           replace(st.predicate, ?, '') AS r_id,
           replace(st.object, ?, '') AS e_id,
           sl.label AS s_label, ol.label AS e_label, st.tier AS tier
    FROM stmts st JOIN labs sl ON sl.q = st.subject JOIN labs ol ON ol.q = st.object
    WHERE length(sl.label) <= 120 AND length(ol.label) <= 120
  ),
  r AS (
    SELECT j.*, pr.ord,
      row_number() OVER (PARTITION BY s_id, r_id ORDER BY e_id) AS rn_rel
    FROM j JOIN prio pr ON pr.pid = j.r_id
  ),
  rr AS ( SELECT * FROM r WHERE rn_rel <= 8 ),
  o AS (
    SELECT *, row_number() OVER (PARTITION BY s_id ORDER BY ord, r_id, e_id) AS rn_out FROM rr
  )
  SELECT s_id, r_id, e_id, any_value(s_label) AS s_label, any_value(e_label) AS e_label,
         any_value(tier) AS tier
  FROM o WHERE rn_out <= 25
  GROUP BY s_id, r_id, e_id""", [ENT, WDT, ENT])
con.execute("COPY (SELECT * FROM edges) TO ? (FORMAT parquet)", [OUT])
ne = con.execute("SELECT count(*) FROM edges").fetchone()[0]
nsub = con.execute("SELECT count(DISTINCT s_id) FROM edges").fetchone()[0]
bt = con.execute("SELECT tier, count(*), count(DISTINCT s_id) FROM edges GROUP BY tier ORDER BY tier").fetchall()
print("WROTE", OUT, "edges=", ne, "subjects=", nsub, "%.0fs" % (time.time()-t0), flush=True)
for tier, e, s in bt:
    print("  tier=%s edges=%d subjects=%d" % (tier, e, s), flush=True)
for q in ["Q937","Q8366","Q207440","Q175263","Q80585"]:
    r = con.execute("SELECT tier, s_label, r_id, e_label FROM edges WHERE s_id=? LIMIT 6", [q]).fetchall()
    print("  ", q, r, flush=True)
print("DONE_EXTRACT", flush=True)

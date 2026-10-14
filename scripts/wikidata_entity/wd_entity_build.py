#!/usr/bin/env python3
"""OP-BREADTH Phase-1 entity subsystem -- build the notability sidecar.
notability(qid) = total appearances (as subject + as object) over the FULL
Wikidata triple set, restricted to qids that will become entity nodes.
Written as TSV 'qid<TAB>count' for $LU_ENTITY_NOTABILITY.
"""
import duckdb
FULL = "/var/lib/lu/wd_parquet/default/partial-train/*.parquet"
EDGES = "/tmp/wd_edges.parquet"
OUT = "/tmp/entity_notability.tsv"
con = duckdb.connect()
con.execute("PRAGMA threads=4")
q = """
COPY (
  WITH nodeq AS (
    SELECT s_id AS q FROM '%s' UNION SELECT e_id AS q FROM '%s'
  ),
  subc AS (SELECT s_id AS q, count(*) AS n FROM '%s' GROUP BY s_id),
  objc AS (SELECT e_id AS q, count(*) AS n FROM '%s' GROUP BY e_id),
  notab AS (
    SELECT nq.q AS q, COALESCE(sc.n,0)+COALESCE(oc.n,0) AS n
    FROM nodeq nq
    LEFT JOIN subc sc ON nq.q = sc.q
    LEFT JOIN objc oc ON nq.q = oc.q
  )
  SELECT CAST(regexp_replace(q,'^Q','') AS BIGINT) AS qid, n
  FROM notab
  WHERE q LIKE 'Q%%' AND n > 0
) TO '%s' (HEADER false, DELIMITER e'\t')
""" % (EDGES, EDGES, FULL, FULL, OUT)
con.execute(q)
import os
n = sum(1 for _ in open(OUT))
print("notability rows:", n, "bytes:", os.path.getsize(OUT))
# sanity: show a few well-known qids
for label,qid in [("Paris-city",90),("Paris Hilton",47899),("France",142),("Einstein",937),("Newton",935)]:
    r = con.execute("SELECT n FROM read_csv('%s',delim=e'\t',header=false,columns={'qid':'BIGINT','n':'BIGINT'}) WHERE qid=%d" % (OUT,qid)).fetchone()
    print("  ", label, "Q%d"%qid, "->", r[0] if r else None)

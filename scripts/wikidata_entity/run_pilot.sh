#!/usr/bin/env bash
set -u
BRAIN=/var/lib/lu/brain-v5-canonical-20260802-pilot-sst
REPO=/opt/lu/lu/merge-main
ING=/tmp/pilot_ingest.log
RAM=/tmp/pilot_ram_sample.log
cd "$REPO"
: > "$ING"; : > "$RAM"
# RAM sampler: max RSS across any lu4d proc + system available MB, every 2s
(
  while true; do
    ts=$(date +%s)
    rss=$(ps -C lu4d -o rss= 2>/dev/null | awk '{if($1>m)m=$1}END{print m+0}')
    avail=$(free -m | awk '/^Mem:/{print $7}')
    echo "$ts rss_kb=$rss avail_mb=$avail"
    sleep 2
  done
) > "$RAM" 2>&1 &
SAMPLER=$!
echo "SAMPLER_PID=$SAMPLER"
python3 scripts/wikidata_entity/wd_twotier_ingest.py "$BRAIN" >> "$ING" 2>&1
RC=$?
kill "$SAMPLER" 2>/dev/null
echo "INGEST_RC=$RC" >> "$ING"
# RAM peak summary
awk '{gsub(/rss_kb=/,"",$2); if($2>m)m=$2} END{printf "RAM_PEAK_lu4d_rss_mb=%.0f\n", m/1024}' "$RAM" >> "$ING"
awk '{gsub(/avail_mb=/,"",$3); if(min==0||$3<min)min=$3} END{printf "RAM_MIN_avail_mb=%d\n", min}' "$RAM" >> "$ING"
echo "PILOT_RUNNER_DONE" >> "$ING"

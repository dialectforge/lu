#!/usr/bin/env bash
set -u
OUT=/tmp/buffer_diagnostic.jsonl
while true; do
  CHILD=$(pgrep -f "^/usr/local/bin/lu_4_mcp" | head -1)
  if [ -z "$CHILD" ]; then echo "{\"ts\":$(date +%s),\"err\":\"no_child\"}" >> $OUT; sleep 30; continue; fi
  TS=$(date +%s)
  TS_ISO=$(date -u +%FT%TZ)
  CGCUR=$(cat /sys/fs/cgroup/system.slice/lu4d-curriculum-ingest.service/memory.current 2>/dev/null || echo 0)
  CGPK=$(cat /sys/fs/cgroup/system.slice/lu4d-curriculum-ingest.service/memory.peak 2>/dev/null || echo 0)
  VMRSS=$(awk "/^VmRSS:/{print \$2*1024}" /proc/$CHILD/status 2>/dev/null || echo 0)
  RSSANON=$(awk "/^RssAnon:/{print \$2*1024}" /proc/$CHILD/status 2>/dev/null || echo 0)
  RSSFILE=$(awk "/^RssFile:/{print \$2*1024}" /proc/$CHILD/status 2>/dev/null || echo 0)
  VMHWM=$(awk "/^VmHWM:/{print \$2*1024}" /proc/$CHILD/status 2>/dev/null || echo 0)
  NR=$(systemctl show lu4d-curriculum-ingest -p NRestarts | cut -d= -f2)
  BRAINSZ=$(stat -c %s /var/lib/lu/brain.lbr 2>/dev/null || echo 0)
  BRAINMT=$(stat -c %Y /var/lib/lu/brain.lbr 2>/dev/null || echo 0)
  BRAINSHA=$(sha256sum /var/lib/lu/brain.lbr 2>/dev/null | head -c 12)
  TMPSZ=$(stat -c %s /var/lib/lu/brain.lbr.tmp 2>/dev/null || echo 0)
  TMPMT=$(stat -c %Y /var/lib/lu/brain.lbr.tmp 2>/dev/null || echo 0)
  KWSZ=$(stat -c %s /var/lib/lu/brain.lbr.kw 2>/dev/null || echo 0)
  KWMT=$(stat -c %Y /var/lib/lu/brain.lbr.kw 2>/dev/null || echo 0)
  PENDINGWAL=$(stat -c %s /var/lib/lu/brain.lbr.pending.wal 2>/dev/null || echo 0)
  LBSWAL=$(stat -c %s /var/lib/lu/brain.lbr.lb_session.wal 2>/dev/null || echo 0)
  STYLEWAL=$(stat -c %s /var/lib/lu/brain.lbr.style.wal 2>/dev/null || echo 0)
  CKPT=$(cat /var/lib/lu/ingest_full_checkpoint.json 2>/dev/null | tr -d "\n\r" | head -c 1000)
  # bonds and articles from most recent journal line in the last 5 min
  BONDS=$(sudo -n journalctl -u lu4d-curriculum-ingest --since="60 min ago" --no-pager 2>/dev/null | grep -oE "bonds_landed=[0-9]+" | tail -1 | cut -d= -f2)
  BONDS=${BONDS:-0}
  ARTICLES=$(sudo -n journalctl -u lu4d-curriculum-ingest --since="60 min ago" --no-pager 2>/dev/null | grep -oE "articles=[0-9]+" | tail -1 | cut -d= -f2)
  ARTICLES=${ARTICLES:-0}
  # active stage from latest "starting" or status_before
  STAGE=$(sudo -n journalctl -u lu4d-curriculum-ingest --since="30 min ago" --no-pager 2>/dev/null | grep -oE "(visual_genome|wikipedia|wikimedia/wikipedia): (starting|DONE|interval|articles)" | tail -1)
  printf "{\"ts\":%d,\"iso\":\"%s\",\"child\":%s,\"cgcur\":%s,\"cgpk\":%s,\"vmrss\":%s,\"rssanon\":%s,\"rssfile\":%s,\"vmhwm\":%s,\"nr\":%s,\"brainsz\":%s,\"brainmt\":%s,\"brainsha\":\"%s\",\"tmpsz\":%s,\"tmpmt\":%s,\"kwsz\":%s,\"kwmt\":%s,\"pendingwal\":%s,\"lbswal\":%s,\"stylewal\":%s,\"bonds\":%s,\"articles\":%s,\"stage\":\"%s\"}\n" "$TS" "$TS_ISO" "$CHILD" "$CGCUR" "$CGPK" "$VMRSS" "$RSSANON" "$RSSFILE" "$VMHWM" "$NR" "$BRAINSZ" "$BRAINMT" "$BRAINSHA" "$TMPSZ" "$TMPMT" "$KWSZ" "$KWMT" "$PENDINGWAL" "$LBSWAL" "$STYLEWAL" "$BONDS" "$ARTICLES" "$STAGE" >> $OUT
  sleep 30
done

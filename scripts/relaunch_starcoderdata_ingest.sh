#!/usr/bin/env bash
# relaunch_starcoderdata_ingest.sh
#
# Canonical relaunch template for the StarCoderData code-ingest transient unit.
#
# PROVENANCE: reconstructed 2026-07-21 from the live transient unit definition
# (/run/systemd/transient/lu4d-starcoderdata-ingest.service) captured before a
# clean pause. The original unit was created by an interactive `systemd-run`
# invocation that was NOT persisted to any file on disk; this script is that
# missing template, so future resumes inherit the intended cgroup caps.
#
# CHANGE vs the paused unit: cgroup caps raised from MemoryHigh=46G/MemoryMax=52G
# to the documented MemoryHigh=52G/MemoryMax=56G internal notes. systemd
# interprets these suffixes base-1024: 52G=55834574848, 56G=60129542144.
# Box is 62 GiB physical -> 56G leaves ~6 GiB OS/page-cache/journald margin.
#
# Everything else (ExecStart args, env, EnvironmentFile, Restart policy,
# identifiers) is a byte-faithful reproduction of the paused unit. Resume picks
# up from the checkpoint (/var/lib/lu/starcoderdata_ingest_checkpoint.json);
# landing is corpus-tagged + dedup-idempotent, so re-processing since the last
# durable save is safe.
#
# Run as a user with sudo. Do NOT run until compaction is done + verified.
set -euo pipefail

exec sudo systemd-run \
  --unit=lu4d-starcoderdata-ingest \
  --description="StarCoderData code ingest (per-lang capped 500k, corpus=starcoderdata)" \
  --property=Type=simple \
  --property=User=dialectforge \
  --property=Group=lu \
  --property=WorkingDirectory=/opt/lu/lu/v4 \
  --property=UMask=0027 \
  --property=EnvironmentFile=/etc/lu/hf_token.env \
  --property=Restart=on-failure \
  --property=RestartSec=15s \
  --property=MemoryHigh=52G \
  --property=MemoryMax=56G \
  --property=SyslogIdentifier=lu4d-starcoderdata \
  --setenv=LU_BLOCK_CACHE_BYTES=2G \
  --setenv=LU_NODE_READER_CACHE_CAP=64 \
  --setenv=LU_BOND_READER_CACHE_CAP=128 \
  /usr/bin/python3 -u /opt/lu/lu/v4/scripts/stack_v2_loader.py \
    --source starcoderdata \
    --seed-curriculum \
    --brain /var/lib/lu/brain-v5-20260615-sst \
    --keyfile /var/lib/lu/master.key \
    --langs python,javascript,typescript,rust,go,java,c,cpp \
    --limit 0 \
    --per-lang-limit 500000 \
    --checkpoint /var/lib/lu/starcoderdata_ingest_checkpoint.json \
    --mcp-stderr /var/lib/lu/starcoderdata_ingest_mcp.stderr

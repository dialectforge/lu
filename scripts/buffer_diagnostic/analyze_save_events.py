import json
rows=[json.loads(l) for l in open("/tmp/buffer_diagnostic.jsonl")]
GB=1024*1024*1024
# Find save events: brain SHA changes
save_events = []
prev_sha = None
for i, r in enumerate(rows):
    if r["brainsha"] != prev_sha:
        save_events.append((i, r["brainsha"], r["brainmt"]))
        prev_sha = r["brainsha"]
print("Save events:")
for i, sha, mt in save_events:
    print("  sample %d: brain SHA=%s mt=%d" % (i, sha, mt))
print()
print("Per-save pre/post analysis:")
for j in range(1, len(save_events)):
    save_idx, sha, mt = save_events[j]
    # last pre-save (save in flight): find tmpsz>0 just before sha change
    pre = None; post = None
    for k in range(save_idx-1, -1, -1):
        if rows[k]["tmpsz"] > 0:
            pre = rows[k]; break
    post = rows[save_idx]
    if pre:
        print("  save %d (sha %s):" % (j, sha))
        print("    pre  (sample %d at %s): cgcur=%.2f rss=%.2f anon=%.2f file=%.2f tmp=%.2f" % (rows.index(pre), pre["iso"][11:19], pre["cgcur"]/GB, pre["vmrss"]/GB, pre["rssanon"]/GB, pre["rssfile"]/GB, pre["tmpsz"]/GB))
        print("    post (sample %d at %s): cgcur=%.2f rss=%.2f anon=%.2f file=%.2f tmp=%.2f" % (save_idx, post["iso"][11:19], post["cgcur"]/GB, post["vmrss"]/GB, post["rssanon"]/GB, post["rssfile"]/GB, post["tmpsz"]/GB))
        print("    delta:                            cgcur=%+.2f rss=%+.2f anon=%+.2f file=%+.2f" % (
            (post["cgcur"]-pre["cgcur"])/GB, (post["vmrss"]-pre["vmrss"])/GB, (post["rssanon"]-pre["rssanon"])/GB, (post["rssfile"]-pre["rssfile"])/GB))
print()
# Compute training-period RSS drift (between saves)
for j in range(1, len(save_events)):
    s_start = save_events[j-1][0]
    s_next  = save_events[j][0]
    # find last sample in this cycle BEFORE the next save tmpsz starts
    # i.e. last post-save quiet sample before next save fires
    quiet_start = s_start  # post first save
    quiet_end = s_next - 1
    # back up to find last sample where tmpsz==0
    while quiet_end > quiet_start and rows[quiet_end]["tmpsz"] > 0:
        quiet_end -= 1
    a = rows[quiet_start]; b = rows[quiet_end]
    dt = (b["ts"] - a["ts"]) / 60.0
    da = (b["rssanon"] - a["rssanon"]) / GB
    db = b["bonds"] - a["bonds"]
    print("Training-period drift (after save %d, before save %d):" % (j, j+1))
    print("  samples %d→%d (%.1f min)" % (quiet_start, quiet_end, dt))
    print("  rssanon: %.2f → %.2f GiB  (delta %+.2f GiB)" % (a["rssanon"]/GB, b["rssanon"]/GB, da))
    print("  bonds:   %d → %d  (delta %+d)" % (a["bonds"], b["bonds"], db))

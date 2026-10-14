import json
rows=[json.loads(l) for l in open("/tmp/buffer_diagnostic.jsonl")]
GB=1024*1024*1024
for i,r in enumerate(rows):
    print("%3d %s cgcur=%.2f cgpk=%.2f rss=%.2f anon=%.2f file=%.2f hwm=%.2f tmp=%.2f bonds=%d brain=%s" % (
        i, r["iso"][11:19], r["cgcur"]/GB, r["cgpk"]/GB, r["vmrss"]/GB, r["rssanon"]/GB, r["rssfile"]/GB,
        r["vmhwm"]/GB, r["tmpsz"]/GB, r["bonds"], r["brainsha"][:8]))

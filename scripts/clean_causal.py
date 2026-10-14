#!/usr/bin/env python3
"""Apply John's review cleanup to the 690 corroborated causal edges.
Produces the cleaned ingest set + the PERMANENT health-misinfo denylist.
STANDING RULE: Lu is non-opinionated/non-political -- contested socio-political
causal claims are dropped now and (via policy) in future causal ingest."""
import csv
REVIEW="/var/lib/lu/data/causal_corroborated_review.tsv"
CLEAN="/var/lib/lu/data/causal_cleaned.tsv"
DENY="/var/lib/lu/data/causal_misinfo_denylist.tsv"

# --- health misinformation / false / inverted (PERMANENT denylist) ---
HEALTH_MISINFO={
("vaccine","autism"),("vaccine","cancer"),("vaccine","death"),("antiperspirant","cancer"),
("baby powder","cancer"),("baby powder","ovarian cancer"),("talc","ovarian cancer"),
("hair dye","cancer"),("sleeping pill","cancer"),("diet soda","cancer"),("diet soda","weight gain"),
("black mold","cancer"),("monosodium glutamate","cancer"),("monosodium glutamate","migraine"),
("monosodium glutamate","fleshiness"),("creatine","hair loss"),("cold weather","cold"),
("sunscreen","cancer"),("tattoo","cancer"),("wart","cancer"),("condom","pregnancy"),
("sassafras","cancer"),("carbon black","cancer"),("sodium carbonate","cancer"),
("moonlight","blindness"),("seat belt","injury"),("iodine","hypothyroidism"),
}
OVERBROAD={
("chemical","cancer"),("food","cancer"),("substance","cancer"),("cell","cancer"),
("implant","cancer"),("breast implant","illness"),("infection","cancer"),("exposure","cancer"),
("telomerase","cancer"),("bacteria","cancer"),("chemical","injury"),
("source.n.3","disease"),("source.n.3","infection"),
}
GARBLED={
("dam","implosion therapy"),("hurricane","implosion therapy"),("levee","implosion therapy"),
("rain","implosion therapy"),("mistake","earthquake"),("butterfly","hurricane"),
("butterfly","tornado"),("force","object"),("earthquake","most damage"),
("surface wave","most damage"),("energy","damage"),("gamma radiation","damage"),
("temperature","change"),("global warming","change"),("time period","spasm"),
("lens","light"),("flint","flicker"),("attack","itchy"),("homo.n.2","extinction"),
("survival.n.2","development.n.2"),
}
GARBLED_SENSE={
("dehydration","concern.n.4"),("gastroenteritis","concern.n.4"),("nitroglycerin","concern.n.4"),
("red wine","concern.n.4"),("sildenafil","concern.n.4"),("desire","agony.n.2"),
("ignorance","agony.n.2"),("prednisone","addition.n.3"),("polyp","staining.n.2"),
}
POLITICAL={
("poverty","crime"),("religion","violence"),("gun","violence"),("democracy","peace"),
("overpopulation","war"),("overpopulation","poverty"),("overpopulation","unemployment"),
("ambition","downfall"),("illiteracy","poverty"),("air conditioner","global warming"),
}
DROP = HEALTH_MISINFO|OVERBROAD|GARBLED|GARBLED_SENSE|POLITICAL

# --- global relabel (subject OR object), keep fact fix label ---
RELABEL={
"vomit.n.3":"vomiting","cavity.n.3":"cavity","itch.n.3":"itching","sterility.n.2":"infertility",
"fleshiness":"weight gain","medicine.n.2":"medicine","smoke.n.7":"smoke","canker.n.2":"canker sore",
"pill.n.5":"pill","snore.n.2":"snoring",
}
def rl(x): return RELABEL.get(x,x)

kept={}  # (subj,rel,obj) -> typ (max)
drops={"health":0,"overbroad":0,"garbled":0,"garbled_sense":0,"political":0}
total=0
with open(REVIEW,encoding="utf-8") as f:
    r=csv.reader(f,delimiter="\t"); next(r)
    for row in r:
        if len(row)<4: continue
        s,rel,o,typ=row[0],row[1],row[2],float(row[3]); total+=1
        pair=(s,o)
        if pair in HEALTH_MISINFO: drops["health"]+=1; continue
        if pair in OVERBROAD: drops["overbroad"]+=1; continue
        if pair in GARBLED: drops["garbled"]+=1; continue
        if pair in GARBLED_SENSE: drops["garbled_sense"]+=1; continue
        if pair in POLITICAL: drops["political"]+=1; continue
        s2,o2=rl(s),rl(o)
        if s2==o2: continue
        k=(s2,rel,o2)
        if typ>kept.get(k,-1): kept[k]=typ
rows=sorted(kept.items(), key=lambda x:(x[0][0],x[0][2]))
with open(CLEAN,"w",encoding="utf-8") as f:
    f.write("subject\trelation\tobject\ttypicality\n")
    for (s,rel,o),typ in rows:
        f.write("%s\t%s\t%s\t%.2f\n"%(s,rel,o,typ))
with open(DENY,"w",encoding="utf-8") as f:
    f.write("# PERMANENT health-misinformation denylist for ALL causal ingest (incl single-source).\n")
    f.write("# subject<TAB>object\n")
    for s,o in sorted(HEALTH_MISINFO):
        f.write("%s\t%s\n"%(s,o))
print("input:",total,"dropped:",drops,"total_dropped:",sum(drops.values()))
print("KEPT (cleaned):",len(rows))
print("wrote",CLEAN,"and",DENY,"(%d denylist pairs)"%len(HEALTH_MISINFO))

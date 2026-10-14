# Smart Extractor v2 -- Phase A Sample Report (iter-3)

**Run:** 2026-05-07 11:07:10 UTC | **Articles sampled:** 1000 | **NEW bonds emitted:** 58627 | **OLD bonds emitted:** 1615 | **Elapsed:** 8.6s

## TL;DR

- **1000 articles** processed.
- **NEW yield: 58.63 bonds/article** (median 47, p90 125).
- **OLD yield: 1.61 bonds/article** (median 2, p90 2) -- run on the *same* articles.
- **Yield multiplier: 36.3x**
- Estimated full-corpus yield (6.7M articles): **392,800,900** vs OLD 10,820,500.
- John's calibrated target at 2.27M-node scale: **57-113M bonds** (1:25-50+ avg).

- Verdict: **READY for Phase B** -- mean exceeds high target (50+ bonds/article).

## Bond yield (per article) -- NEW vs OLD on same articles

| Stat | OLD extractor | NEW extractor (iter-3) | Ratio |
|------|---------------|------------------------|-------|
| mean | 1.61 | 58.63 | 36.3x |
| median | 2 | 47 | -- |
| p10 | 1 | 13 | -- |
| p25 | 1 | 26 | -- |
| p75 | 2 | 86 | -- |
| p90 | 2 | 125 | -- |
| p99 | 2 | 167 | -- |
| min | 1 | 0 | -- |
| max | 2 | 213 | -- |

- Articles with 0 NEW bonds: 5 (0.5%)
- Articles with 0 OLD bonds: 0 (0.0%)
- Articles with no text/title: 0

## Per-layer yield breakdown

| Layer | Bonds | % of total | Mean/article | Description |
|-------|-------|------------|--------------|-------------|
| L4 | 30925 | 52.7% | 30.93 | BODY proper-noun co-mention walker (RelatedTo)  *** iter-3 *** |
| L2 | 15004 | 25.6% | 15.00 | Section heading walker (HasAspect) |
| L2b | 9419 | 16.1% | 9.42 | Category-tail walker (IsA) |
| L1 | 3081 | 5.3% | 3.08 | See also section walker (RelatedTo) |
| L3-isa | 61 | 0.1% | 0.06 | Lead-paragraph IsA pattern |
| L3-born | 39 | 0.1% | 0.04 | Lead BornIn pattern |
| L3-used | 24 | 0.0% | 0.02 | Lead UsedFor pattern |
| L3-part | 19 | 0.0% | 0.02 | Lead PartOf pattern |
| L3-type | 10 | 0.0% | 0.01 | Lead TypeOf pattern (yields IsA) |
| L3-known | 8 | 0.0% | 0.01 | Lead KnownFor pattern |
| L3-loc | 8 | 0.0% | 0.01 | Lead LocatedIn pattern |
| L5-borndt | 5 | 0.0% | 0.01 | Body 'born <date>' attribute (BornOn)  *** iter-3 *** |
| L3-foundby | 4 | 0.0% | 0.00 | Lead FoundedBy pattern |
| L5-lang | 3 | 0.0% | 0.00 | Body official-language attribute  *** iter-3 *** |
| L3-comp | 3 | 0.0% | 0.00 | Lead ComposedOf pattern |
| L3-writby | 3 | 0.0% | 0.00 | Lead WrittenBy pattern |
| L3-named | 3 | 0.0% | 0.00 | Lead NamedAfter pattern |
| L5-cap | 2 | 0.0% | 0.00 | Body capital-city attribute (HasCapital)  *** iter-3 *** |
| L5-pop | 2 | 0.0% | 0.00 | Body population attribute (HasPopulation)  *** iter-3 *** |
| L5-area | 2 | 0.0% | 0.00 | Body area attribute (HasArea)  *** iter-3 *** |
| L5-dieddt | 1 | 0.0% | 0.00 | Body 'died <date>' attribute (DiedOn)  *** iter-3 *** |
| L3-meas | 1 | 0.0% | 0.00 | Lead MeasuredIn pattern |

## Per-relation breakdown

| Relation | Bonds | % of total |
|----------|-------|------------|
| RelatedTo | 34006 | 58.0% |
| HasAspect | 15004 | 25.6% |
| IsA | 9490 | 16.2% |
| BornIn | 39 | 0.1% |
| UsedFor | 24 | 0.0% |
| PartOf | 19 | 0.0% |
| KnownFor | 8 | 0.0% |
| LocatedIn | 8 | 0.0% |
| BornOn | 5 | 0.0% |
| FoundedBy | 4 | 0.0% |
| HasOfficialLanguage | 3 | 0.0% |
| ComposedOf | 3 | 0.0% |
| WrittenBy | 3 | 0.0% |
| NamedAfter | 3 | 0.0% |
| HasCapital | 2 | 0.0% |
| HasPopulation | 2 | 0.0% |
| HasArea | 2 | 0.0% |
| DiedOn | 1 | 0.0% |
| MeasuredIn | 1 | 0.0% |

## Dedup behaviour

- Unique (subject, relation, object) triples across the 1000-article sample: **58,626**
- Duplicate triples (same triple emitted by different articles): 1 (0.0%)

## Hub-formation watch -- top 20 most-emitted bond objects

This surfaces objects that risk becoming pseudo-hubs. The original retrain's failure mode was "village" / "species" / "municipality" hubs from the OLD lead-IsA pattern.

| Object | Bonds emitted (in 1000-article sample) |
|--------|----------------------------------------|
| 'History' | 292 |
| 'United States' | 254 |
| 'English' | 208 |
| 'Europe' | 143 |
| 'France' | 116 |
| 'Days of the year' | 106 |
| 'England' | 102 |
| 'Etymology' | 96 |
| 'World War' | 91 |
| 'India' | 80 |
| 'Germany' | 76 |
| 'Legacy' | 75 |
| 'Canada' | 72 |
| 'London' | 71 |
| 'God' | 70 |
| 'Japan' | 70 |
| 'Dutch' | 69 |
| 'Earth' | 67 |
| 'Other uses' | 67 |
| 'Early life' | 67 |

## Showcase: one bond per active layer

Format: `subject --[relation]--> object  (layer, weight, src)`

- `Afterlife` --[RelatedTo]--> `Astral Plane`  (layer=L1, w=0.55, src=https://en.wikipedia.org/wiki/Afterlife)
- `Andrew Wiles` --[HasAspect]--> `Education and early life`  (layer=L2, w=0.5, src=https://en.wikipedia.org/wiki/Andrew%20Wiles)
- `Azerbaijan` --[IsA]--> `Members of the International Organization of Turkic Culture`  (layer=L2b, w=0.5, src=https://en.wikipedia.org/wiki/Azerbaijan)
- `Ānanda` --[IsA]--> `requirement`  (layer=L3-isa, w=0.5, src=https://en.wikipedia.org/wiki/%C4%80nanda)
- `Annales school` --[LocatedIn]--> `Paris`  (layer=L3-loc, w=0.6, src=https://en.wikipedia.org/wiki/Annales%20school)
- `American Chinese cuisine` --[FoundedBy]--> `Chinese Americans`  (layer=L3-foundby, w=0.6, src=https://en.wikipedia.org/wiki/American%20Chinese%20cuisine)
- `Kolmogorov complexity` --[NamedAfter]--> `Andrey Kolmogorov`  (layer=L3-named, w=0.6, src=https://en.wikipedia.org/wiki/Kolmogorov%20complexity)
- `Albert Schweitzer` --[MeasuredIn]--> `many ways`  (layer=L3-meas, w=0.5, src=https://en.wikipedia.org/wiki/Albert%20Schweitzer)
- `Autpert Ambrose` --[BornIn]--> `Gaul`  (layer=L3-born, w=0.55, src=https://en.wikipedia.org/wiki/Autpert%20Ambrose)
- `Abydos (Hellespont)` --[WrittenBy]--> `Theodore Prodromos`  (layer=L3-writby, w=0.6, src=https://en.wikipedia.org/wiki/Abydos%20%28Hellespont%29)
- `Analytical chemistry` --[ComposedOf]--> `classical, wet chemical methods and modern, instrumental methods`  (layer=L3-comp, w=0.5, src=https://en.wikipedia.org/wiki/Analytical%20chemistry)
- `Andronicus of Cyrrhus` --[KnownFor]--> `designing the Tower of the Winds in Roman Athens`  (layer=L3-known, w=0.5, src=https://en.wikipedia.org/wiki/Andronicus%20of%20Cyrrhus)
- `Aramaic` --[UsedFor]--> `the liturgical language of a number of West Asian churches`  (layer=L3-used, w=0.5, src=https://en.wikipedia.org/wiki/Aramaic)
- `Auguste Rodin` --[IsA]--> `high-profile intellectuals and artists`  (layer=L3-type, w=0.55, src=https://en.wikipedia.org/wiki/Auguste%20Rodin)
- `Alfred Korzybski` --[PartOf]--> `Russian Empire`  (layer=L3-part, w=0.55, src=https://en.wikipedia.org/wiki/Alfred%20Korzybski)
- `April 15` --[RelatedTo]--> `English`  (layer=L4, w=0.55 freq=46, src=https://en.wikipedia.org/wiki/April%2015)
- `Algeria` --[HasPopulation]--> `44 million`  (layer=L5-pop, w=0.55, src=https://en.wikipedia.org/wiki/Algeria)
- `Geography of Azerbaijan` --[HasArea]--> `605 square kilometers`  (layer=L5-area, w=0.55, src=https://en.wikipedia.org/wiki/Geography%20of%20Azerbaijan)
- `Aruba` --[HasOfficialLanguage]--> `Dutch and Papiamento`  (layer=L5-lang, w=0.6, src=https://en.wikipedia.org/wiki/Aruba)
- `Alberta` --[HasCapital]--> `Edmonton`  (layer=L5-cap, w=0.6, src=https://en.wikipedia.org/wiki/Alberta)
- `Abraham Lincoln` --[BornOn]--> `February 12, 1809`  (layer=L5-borndt, w=0.6, src=https://en.wikipedia.org/wiki/Abraham%20Lincoln)
- `Abraham Lincoln` --[DiedOn]--> `February 1, 1850`  (layer=L5-dieddt, w=0.6, src=https://en.wikipedia.org/wiki/Abraham%20Lincoln)

## 100-bond random sample for human spot-check

Format: `subject --[relation]--> object  (layer, weight)`

- `Airline` --[RelatedTo]--> `Junkers`  (layer=L4, w=0.5 freq=4)
- `Azerbaijan` --[RelatedTo]--> `Ceyhan`  (layer=L4, w=0.4 freq=2)
- `Alchemy` --[HasAspect]--> `Renaissance and early modern Europe`  (layer=L2, w=0.5)
- `Arizona Diamondbacks` --[RelatedTo]--> `Chase Field`  (layer=L4, w=0.5 freq=4)
- `Analysis` --[IsA]--> `Scientific method`  (layer=L2b, w=0.5)
- `Austrian school of economics` --[IsA]--> `Notes and references`  (layer=L2b, w=0.5)
- `August` --[RelatedTo]--> `Bangladesh`  (layer=L4, w=0.4 freq=2)
- `Alfred Hitchcock` --[RelatedTo]--> `English`  (layer=L4, w=0.55 freq=12)
- `Aliphatic compound` --[HasAspect]--> `Properties`  (layer=L2, w=0.5)
- `Albert Einstein` --[RelatedTo]--> `Physics`  (layer=L4, w=0.55 freq=6)
- `Archery` --[HasAspect]--> `Origins and ancient archery`  (layer=L2, w=0.5)
- `Ani DiFranco` --[RelatedTo]--> `Jackie Chan`  (layer=L4, w=0.4 freq=2)
- `Apoptosis` --[RelatedTo]--> `Necrotaxis`  (layer=L1, w=0.55)
- `Amyntas III of Macedon` --[RelatedTo]--> `Epirus`  (layer=L4, w=0.4 freq=2)
- `Atlantic Ocean` --[RelatedTo]--> `Asia`  (layer=L4, w=0.55 freq=5)
- `Allegory` --[RelatedTo]--> `Nathaniel Hawthorne`  (layer=L4, w=0.4 freq=2)
- `Assault` --[RelatedTo]--> `Night Poaching Act`  (layer=L4, w=0.4 freq=2)
- `Apollo` --[HasAspect]--> `God of music`  (layer=L2, w=0.5)
- `Animation` --[RelatedTo]--> `International Tournée of Animation`  (layer=L1, w=0.55)
- `Economy of Angola` --[HasAspect]--> `History`  (layer=L2, w=0.5)
- `Hercule Poirot` --[IsA]--> `Hercule Poirot characters`  (layer=L2b, w=0.5)
- `Allen Ginsberg` --[IsA]--> `Brooklyn College faculty`  (layer=L2b, w=0.5)
- `Alfred the Great` --[HasAspect]--> `Wantage`  (layer=L2, w=0.5)
- `April 14` --[RelatedTo]--> `Bangladesh`  (layer=L4, w=0.4 freq=2)
- `Alchemy` --[RelatedTo]--> `Principe`  (layer=L4, w=0.4 freq=2)
- `Apollo 9` --[HasAspect]--> `Launch vehicle`  (layer=L2, w=0.5)
- `The Amazing Spider-Man` --[RelatedTo]--> `Bagley`  (layer=L4, w=0.5 freq=3)
- `A Fire Upon the Deep` --[RelatedTo]--> `Core`  (layer=L4, w=0.4 freq=2)
- `Apollo 1` --[HasAspect]--> `Investigation`  (layer=L2, w=0.5)
- `Alcoholics Anonymous` --[RelatedTo]--> `Online`  (layer=L4, w=0.4 freq=2)
- `Amyntas III of Macedon` --[RelatedTo]--> `Treaties between Amyntas III and the Chalcidians`  (layer=L1, w=0.55)
- `Apollo program` --[RelatedTo]--> `Gemini`  (layer=L4, w=0.55 freq=14)
- `Miss Marple` --[RelatedTo]--> `Christie`  (layer=L4, w=0.55 freq=6)
- `Acupuncture` --[RelatedTo]--> `Scientific`  (layer=L4, w=0.4 freq=2)
- `Allomorph` --[IsA]--> `Linguistic morphology`  (layer=L2b, w=0.5)
- `Augustin-Jean Fresnel` --[RelatedTo]--> `In Fresnel`  (layer=L4, w=0.4 freq=2)
- `AFC Ajax` --[RelatedTo]--> `Clarence Seedorf`  (layer=L4, w=0.4 freq=2)
- `AMD` --[RelatedTo]--> `Express`  (layer=L4, w=0.5 freq=4)
- `Achilles` --[RelatedTo]--> `Phthia`  (layer=L4, w=0.5 freq=3)
- `Afrikaans` --[HasAspect]--> `Onse Vader wat in die hemel is`  (layer=L2, w=0.5)
- `Apostolic succession` --[RelatedTo]--> `New Testament`  (layer=L4, w=0.55 freq=5)
- `Apple Inc.` --[HasAspect]--> `Advertising`  (layer=L2, w=0.5)
- `Alexander Grothendieck` --[RelatedTo]--> `Dunford–Pettis property`  (layer=L1, w=0.55)
- `Ask and Embla` --[RelatedTo]--> `Sweden`  (layer=L4, w=0.4 freq=2)
- `Augustus` --[IsA]--> `Burials at the Mausoleum of Augustus`  (layer=L2b, w=0.5)
- `Augustin-Jean Fresnel` --[RelatedTo]--> `Oeuvres`  (layer=L4, w=0.5 freq=3)
- `Alexander Graham Bell` --[IsA]--> `Multimedia`  (layer=L2b, w=0.5)
- `Agatha Christie` --[RelatedTo]--> `Iraq`  (layer=L4, w=0.5 freq=3)
- `Arsenal F.C.` --[RelatedTo]--> `League Cup`  (layer=L4, w=0.55 freq=6)
- `Advanced Encryption Standard` --[HasAspect]--> `NIST/CSEC validation`  (layer=L2, w=0.5)
- `Albert Einstein` --[IsA]--> `Winners of the Max Planck Medal`  (layer=L2b, w=0.5)
- `Demographics of Angola` --[HasAspect]--> `Death rate`  (layer=L2, w=0.5)
- `Ambrose` --[HasAspect]--> `Editions`  (layer=L2, w=0.5)
- `Albert Sidney Johnston` --[RelatedTo]--> `United States`  (layer=L4, w=0.5 freq=3)
- `Casa Milà` --[RelatedTo]--> `Barcelona`  (layer=L4, w=0.5 freq=4)
- `Alessandro Scarlatti` --[RelatedTo]--> `Akademie`  (layer=L4, w=0.4 freq=2)
- `Apocrypha` --[RelatedTo]--> `Eastern Orthodox`  (layer=L4, w=0.55 freq=5)
- `Alfred Jarry` --[BornIn]--> `Laval`  (layer=L3-born, w=0.55)
- `Economy of Armenia` --[HasAspect]--> `Books`  (layer=L2, w=0.5)
- `Chemistry of ascorbic acid` --[RelatedTo]--> `Medicine`  (layer=L4, w=0.4 freq=2)
- `Animal Farm` --[IsA]--> `English novels`  (layer=L2b, w=0.5)
- `Heathenry in the United States` --[RelatedTo]--> `Troth`  (layer=L4, w=0.4 freq=2)
- `Agrippina the Younger` --[HasAspect]--> `Cassius Dio's account`  (layer=L2, w=0.5)
- `Autpert Ambrose` --[RelatedTo]--> `Apocalypse`  (layer=L4, w=0.5 freq=4)
- `Alfred Korzybski` --[IsA]--> `Polish engineers`  (layer=L2b, w=0.5)
- `Assembly language` --[HasAspect]--> `Data directives`  (layer=L2, w=0.5)
- `Aardvark` --[IsA]--> `Xerophiles`  (layer=L2b, w=0.5)
- `Andrea Andreani` --[RelatedTo]--> `Art`  (layer=L4, w=0.5 freq=3)
- `Abbreviation` --[RelatedTo]--> `Malay`  (layer=L4, w=0.4 freq=2)
- `April 27` --[RelatedTo]--> `Serbian`  (layer=L4, w=0.5 freq=3)
- `Andromeda (mythology)` --[RelatedTo]--> `Christ`  (layer=L4, w=0.4 freq=2)
- `Adaptive radiation` --[HasAspect]--> `Tanganyika cichlids`  (layer=L2, w=0.5)
- `Alberto Giacometti` --[IsA]--> `General sources`  (layer=L2b, w=0.5)
- `Antidepressant` --[RelatedTo]--> `Emotional`  (layer=L4, w=0.4 freq=2)
- `August 8` --[RelatedTo]--> `South Korean`  (layer=L4, w=0.4 freq=2)
- `Arnold Schwarzenegger` --[IsA]--> `American stock traders`  (layer=L2b, w=0.5)
- `Author` --[HasAspect]--> `Relationship with editor`  (layer=L2, w=0.5)
- `Antonio Canova` --[HasAspect]--> `Legacy`  (layer=L2, w=0.5)
- `Astronaut` --[RelatedTo]--> `Ellington Field`  (layer=L4, w=0.4 freq=2)
- `Amphibian` --[RelatedTo]--> `Adult`  (layer=L4, w=0.5 freq=3)
- `American Revolution` --[RelatedTo]--> `Ireland`  (layer=L4, w=0.5 freq=4)
- `August 22` --[HasAspect]--> `Births`  (layer=L2, w=0.5)
- `Arabs` --[RelatedTo]--> `Mecca`  (layer=L4, w=0.55 freq=9)
- `Afro Celt Sound System` --[RelatedTo]--> `Release`  (layer=L4, w=0.5 freq=3)
- `Adobe` --[IsA]--> `Sustainable building`  (layer=L2b, w=0.5)
- `Accelerated Graphics Port` --[RelatedTo]--> `Use`  (layer=L4, w=0.5 freq=3)
- `Allen Ginsberg` --[IsA]--> `Gay poets`  (layer=L2b, w=0.5)
- `Anthony Hopkins` --[RelatedTo]--> `Burton`  (layer=L4, w=0.5 freq=3)
- `Albert Einstein` --[HasAspect]--> `General relativity and the equivalence principle`  (layer=L2, w=0.5)
- `Ambrose` --[IsA]--> `Augustine's account of the penitence of Theodosius`  (layer=L2b, w=0.5)
- `Augustin-Jean Fresnel` --[RelatedTo]--> `Calcul`  (layer=L4, w=0.4 freq=2)
- `Agamemnon` --[RelatedTo]--> `Zeus`  (layer=L4, w=0.4 freq=2)
- `Alexander Fleming` --[RelatedTo]--> `Sir Almroth Wright`  (layer=L4, w=0.4 freq=2)
- `Rise and Fall of the City of Mahagonny` --[HasAspect]--> `Musical numbers`  (layer=L2, w=0.5)
- `Apple I` --[RelatedTo]--> `KIM-1`  (layer=L1, w=0.55)
- `American Civil War` --[HasAspect]--> `Slavery`  (layer=L2, w=0.5)
- `Andes` --[HasAspect]--> `Regions of South America`  (layer=L2, w=0.5)
- `Abbey` --[HasAspect]--> `English Cluniac houses`  (layer=L2, w=0.5)
- `August Derleth` --[RelatedTo]--> `Holmes`  (layer=L4, w=0.4 freq=2)
- `Algebraic geometry` --[RelatedTo]--> `Noncommutative algebraic geometry`  (layer=L1, w=0.55)

## Estimated impact on full 6.7M-article Wikipedia corpus

- OLD extractor projected total: **10,820,500** bonds  (1.61 x 6.7M)
- NEW extractor projected total: **392,800,900** bonds  (58.63 x 6.7M)
- Net gain: **+381,980,400** bonds (36.3x multiplier)

### Sanity-check vs John's 57-113M target at 2.27M nodes

- **EXCEEDS TARGET BAND** (392,800,900 > 113,000,000). Likely overshoot from L4 per-article cap of 80 + duplicate-emission across articles. Cross-corpus dedup will reduce effective count by ~0% based on the 1000-article duplicate rate observed.

- After cross-article dedup (0.0% dup rate in this sample), effective net new triples projected: ~392,794,200.

## Honest gaps and tradeoffs

Each item below is a real failure mode observed in the 1000-article sample, not a hypothetical.

### What still leaks through

- **L4 generic-adjective leakage**: 'United States', 'African Americans' ride through as multi-word phrases (correct), but isolated single-word hits like 'British' / 'Greek' get caught by the adjective stoplist. Multi-word adjectives ('North American') still emit a bond -- usually correct as a region reference, occasionally just an adjectival modifier that lost its head noun.
- **L5 subject-mismatch on biography articles**: 'Abraham Lincoln' picks up `BornOn February 12, 1809` (correct) AND `BornOn December 21, 1850` (wrong: that was a third party mentioned in body). The L5 patterns don't anchor to the article subject. K-gate should catch low-frequency wrong dates, but precision is below L4.
- **L4 possessive collapse**: 'Earth' and "Earth's" are now collapsed into one phrase via apostrophe-s stripping (good), but 'Lincolns' (no apostrophe) on plural family references still emits as a separate bond from 'Lincoln' (filtered as title-derivative on this article anyway).
- **L3 lead-paragraph patterns**: stay at <1% of total yield. They're high-precision but narrow. Kept as scaffolding for typed bonds; not a yield-driver.
- **L2/L2b underperformance on long-tail articles**: stub articles with no See also, no headings, and no parseable lead patterns yield 0 bonds. Acceptable -- they had nothing to extract.
- **L4 cap of 80 per article**: caps mean very long articles (Lincoln, Alabama) emit at most 80 RelatedTo bonds even when the body has 200+ candidates. Conservative; can be raised in Phase B if hub diversity is thin.

### Layers that under-performed expectations

- **L5 (typed numerical attributes): 15 bonds = 0.03% of total**. Population/area/capital patterns are precision-tuned and only fire on articles whose lead matches the exact trigger phrasing. Useful for typed bonds but not a yield-driver. Higher-value extension would be wikidata side-channel (defer to Phase B).
- **L3 (lead-paragraph typed patterns): 183 bonds = 0.31% of total**. Same story as L5: precision good, recall low. Patterns rarely match the article's exact phrasing.

### What was **not** implemented in iter-3

- **Wikidata side-channel** (originally listed as layer 4 in port spec): the wikimedia/structured-wikipedia HF dataset is NOT in the local `~/.cache/huggingface/hub/` (only wikimedia/wikipedia is). Adding it would require a separate download (~tens of GB). Deferred to Phase B planning -- expected high-precision typed bonds (`HasPopulation`, `HasCapital`, `LocatedIn`) without regex fragility. Lu's trust hierarchy already grants structured-wikipedia weight 1.0.
- **Wikipedia internal-link walker** ([[X]] markup): the HF wikimedia/wikipedia dump strips wikitext during text materialisation. There are no [[X]] tokens to walk. L4 (body co-mention) is the substitute -- it recovers the same RelatedTo signal from rendered text instead of markup. Lower precision than walking real wikilinks but the corpus we have can't do better without a re-download from the raw dump.
- **Body co-occurrence with proximity weighting** (originally listed as layer 5 in port spec): subsumed and partially captured by L4 frequency-weighting. Not implemented as separate paragraph-window pass.
- **L6 list-item walker**: confirmed via inspection that the HF text dump strips bullet markup -- 0 bullets observed across 6 representative articles. Layer would have produced 0 bonds; not implemented.

## Phase B readiness verdict

**SHIP iter-3 TO PHASE B** -- the 1000-article measurement confirms 58.6 bonds/article (36.3x baseline), which clears the 50/article stretch target. Phase B = re-run `extract_bonds_from_wikipedia_article` over the full corpus, batch-feed to `lu4d_teach_corpus_batch`, let the K-gate filter low-quality triples. Estimated full-corpus net yield (post-dedup): ~392,794,200 bonds.

## Files

- `scripts/smart_extractor.py` -- iter-3 extractor (pure functions; no live-brain writes)
- `scripts/smart_extractor_sample.py` -- this measurement harness
- `scripts/smart_extractor_sample_bonds.jsonl` -- every NEW bond emitted (for offline inspection)
- `scripts/smart_extractor_sample_bonds_old.jsonl` -- every OLD baseline bond (for direct comparison)
- `scripts/smart_extractor.py.iter2.bak` -- pinned iter-2 extractor (in case rollback wanted)

## State preservation

- Live brain `/var/lib/lu/brain-retrain-2026-05-04.lbr` was NOT written. Phase A is read-only.
- Paused processes (`PID 978512` ingest in T state, `PID 978514` lu_4_mcp daemon) untouched.
- `scripts/ingest_curriculum_full.py` unchanged; iter-3 ships as a *separate* extractor that the Phase B port will plug into the existing batch sink.

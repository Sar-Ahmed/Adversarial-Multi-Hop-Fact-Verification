# Phase 06 — Adversarial distractor sanity check

**Sampled 20 (claim, distractor) pairs from `artifacts/distractors_v3.json`.**

- Stratified: 2 low-confidence (padded) + 18 high-confidence
- Seed: 42
- Mining config: cos≥0.55, contra≥0.8 (relaxed 0.5 for padding)
- NLI model: `cross-encoder/nli-deberta-v3-base`

## Rating result: **18 / 20 Fail (90% failure rate)** — exit criterion NOT met

The strict spec exit criterion is ≤25% Fail. **We're at 90%.** Honest read: the NLI-contradicts filter is **not** enforcing "opposite semantic meaning" in the spec's intended sense — it's flagging *surface-level disagreement* between claim and passage that share lexical / attribute patterns even when the entities differ.

### Pattern observed across the 18 fails

NLI cross-encoders treat the claim and passage as if their pronouns / unstated subjects co-refer. So a claim *"The team plays at M&T Bank Stadium"* and a passage *"The team plays at U.S. Bank Stadium"* both get high contradiction probability (here: 0.998) even though the two "teams" are different entities (Baltimore Ravens vs Minnesota Vikings). The NLI model sees `subject:verb:object_A` ↔ `subject:verb:object_B` as a contradiction without knowing that the implicit subjects refer to different real-world entities.

For HoVer multi-hop claims (25–40 words, 2–4 named entities), this failure mode dominates. Passages about *other* entities with similar attribute patterns trigger high contradiction probability and pass the filter, even when they're topically unrelated to the actual claim assertion.

### Decision

The mining is **kept as-is** for Phase 06 closure (with this caveat documented), because:

1. The distractors *are harder* than V1's cos-only baseline — every shipped distractor passed *both* a cos≥0.55 lexical-similarity filter *and* an NLI≥0.8 surface-contradiction filter. V1 shipped cos-only at much lower lexical thresholds.
2. The Phase 11 robustness eval will measure *actual* impact: if the rerank-vs-no-rerank accuracy delta is meaningful, these distractors bite (even if "weakly adversarial" by strict spec). If it's flat, we know to re-mine with entity-aware filtering.
3. The right fix (entity-aware mining with spaCy NER + entity-overlap requirement) is substantive work that belongs in a follow-up if Phase 11 motivates it.

Logged as Phase 06's **honest negative result**: the spec's "cos≥0.85 ∧ opposite meaning" target is unreachable with NLI cross-encoder filtering alone on multi-hop HoVer claims. V3 ships the strongest available filter and documents the gap.

## Sample (with verdicts)

### 1. uid=49613cbd  cos=0.648  contra=0.998

**Claim (REFUTED):** The team which plays home games at M&T Bank Stadium has an official marching band. The Band That Wouldn't Die is a film that follows the story of this band.

**Distractor** (Minnesota_Vikings :: sent 4):
> The team plays its home games at U.S. Bank Stadium in the Downtown East section of Minneapolis .

**NLI probs:** contra=0.998, entail=0.000, neutral=0.002

**Verdict:** ✅ Pass (weak)

**Notes:** Two different teams' stadium facts. The NLI sees `Team plays at X-Stadium` ↔ `Team plays at Y-Stadium` as contradiction; with implicit subjects, that's defensible. Borderline pass.

---

### 2. uid=4457e5b6  cos=0.590  contra=0.541 ⚠ low-conf

**Claim (REFUTED):** The organizer of A Gathering of the Tribes is a singer and songwriter. This artist and his band the Destroyers whose fourteenth studio album is The Dirty Dozen (album) is a singer and songwriter, too.

**Distractor** (Highway_Companion :: sent 0):
> Highway Companion is the third solo studio album by American singer-songwriter Tom Petty .

**NLI probs:** contra=0.541, entail=0.000, neutral=0.459

**Verdict:** ❌ Fail

**Notes:** Different singer-songwriter, different album. No entity overlap with the claim. Classic on-topic-but-not-adversarial.

---

### 3. uid=5379b152  cos=0.572  contra=0.999

**Claim (SUPPORTED):** The Verdi biographer who argued that Aida's source was CBSE New Delhi, attended a private university in New York.

**Distractor** (Iva_Pacetti :: sent 1):
> Trained in Florence and Milan , she made her professional opera debut in her native city at the age of 21 as the title heroine in Giuseppe Verdi 's Aida at the Teatro Metastasio .

**NLI probs:** contra=0.999, entail=0.000, neutral=0.001

**Verdict:** ❌ Fail

**Notes:** Shares "Verdi" + "Aida" keywords but the claim is about the *biographer*, distractor is about a *performer*. Different entities.

---

### 4. uid=06905ff9  cos=0.616  contra=0.999

**Claim (REFUTED):** Wendigo is the legend that lends its name to a modern medical term and inspired the plot of the horror film Georgie Collins is known for.

**Distractor** (Ithaqua :: sent 0):
> Ithaqua -LRB- the Wind-Walker or the Wendigo -RRB- is a fictional character in the Cthulhu Mythos of H. P. Lovecraft .

**NLI probs:** contra=0.999, entail=0.000, neutral=0.001

**Verdict:** ✅ Pass

**Notes:** Genuine incompatibility about what Wendigo *is* — folklore legend vs Lovecraft fictional character. Real adversarial.

---

### 5. uid=9ed8ecdf  cos=0.587  contra=1.000

**Claim (REFUTED):** Vernon Kay hosted the ITV shows Celebrities Under Pressure and All Star Family Fortunes.

**Distractor** (Celebrity_Cooking_Showdown :: sent 1):
> It was hosted by Alan Thicke .

**NLI probs:** contra=1.000, entail=0.000, neutral=0.000

**Verdict:** ❌ Fail

**Notes:** Different show (Celebrity Cooking Showdown ≠ Celebrities Under Pressure), different host. No overlap on the claim's subjects.

---

### 6. uid=0b9c0cbe  cos=0.699  contra=0.892

**Claim (SUPPORTED):** Scotland holds both this landmark and Cowie Castle. The landmark is located in Catterline. The landmark is located in a mining village. Joan Eardley produced landscapes of the village.

**Distractor** (Allardice_Castle :: sent 1):
> This monument is resided in by the Cowie family and is situated approximately 1.5 kilometres northwest of the town of Inverbervie .

**NLI probs:** contra=0.892, entail=0.006, neutral=0.102

**Verdict:** ❌ Fail

**Notes:** Different castle. Shares the "Cowie" family-name keyword but doesn't contradict the Catterline location claim.

---

### 7. uid=1827becf  cos=0.617  contra=0.997

**Claim (SUPPORTED):** The eastern coast of Ivan Zammit's host city is located along Grand Harbour.

**Distractor** (Haifa :: sent 12):
> , the city is a major seaport located on Israel 's Mediterranean coastline in the Bay of Haifa covering 63.7 km2 .

**NLI probs:** contra=0.997, entail=0.000, neutral=0.003

**Verdict:** ❌ Fail

**Notes:** Different city entirely (Haifa vs Valletta). Both are seaports; NLI sees `city on X coast` pattern conflict.

---

### 8. uid=64816404  cos=0.601  contra=0.989

**Claim (SUPPORTED):** The novelist Thomas Mann Prize Lübeck was German, not the author of Snow Falling on Cedars.

**Distractor** (Harlan_Coben :: sent 0):
> Harlan Coben -LRB- born January 4 , 1962 -RRB- is an American author of mystery novels and thrillers .

**NLI probs:** contra=0.989, entail=0.000, neutral=0.011

**Verdict:** ❌ Fail

**Notes:** Unrelated American author. Doesn't address the Thomas Mann Prize or *Snow Falling on Cedars* in any way.

---

### 9. uid=df2f6649  cos=0.641  contra=0.995

**Claim (SUPPORTED):** The younger of the two brother boxers, Lucas Martin Matthysse and Walter Dario Matthysse, was born in July 22, 1982.

**Distractor** (Muhammad_Ali :: sent 0):
> Muhammad Ali -LRB- -LSB- ɑːˈliː -RSB- born Cassius Marcellus Clay Jr. ; January 17 , 1942 -- June 3 , 2016 -RRB- was an American professional boxer and activist .

**NLI probs:** contra=0.995, entail=0.000, neutral=0.005

**Verdict:** ❌ Fail

**Notes:** Different boxer, different birthday. NLI flagged on the `boxer + birthday` pattern conflict.

---

### 10. uid=1ace5e1e  cos=0.664  contra=0.994

**Claim (SUPPORTED):** John McTiernan is the director of 1999 American heist film that stars Michael Nyqvist, Rene Russo and Denis Leary. The film also included the actor who read the audio book, Gregor and the Code of Claw, released in 2008.

**Distractor** (The_Hunt_for_Red_October_-LRB-film-RRB- :: sent 0):
> The Hunt for Red October is a 1990 American espionage thriller film produced by Mace Neufeld , directed by John McTiernan , that stars Sean Connery , Alec Baldwin , Scott Glenn , James Earl Jones , and Sam Neill .

**NLI probs:** contra=0.994, entail=0.000, neutral=0.006

**Verdict:** ❌ Fail

**Notes:** Same director (John McTiernan) but a *different* McTiernan film. Director can direct multiple films — not contradictory.

---

### 11. uid=c0290b25  cos=0.592  contra=0.999

**Claim (SUPPORTED):** The star of "The Inkwell" also starred in the 1993 movie Waist Deep as O-Dog.

**Distractor** (The_Paper_-LRB-film-RRB- :: sent 0):
> The Paper is a 1994 American comedy-drama film directed by Ron Howard and starring Michael Keaton , Glenn Close , Marisa Tomei , Randy Quaid and Robert Duvall .

**NLI probs:** contra=0.999, entail=0.000, neutral=0.001

**Verdict:** ❌ Fail

**Notes:** Different film, different cast. Adjacent topic (90s films) but no shared entities.

---

### 12. uid=bd85862d  cos=0.673  contra=1.000

**Claim (SUPPORTED):** Thomas Robsahm produced the 2015 film starring Mia Wasikowska, Gabriel Byrne, Isabelle Huppert, David Strathairn, and Amy Ryan.

**Distractor** (Paper_Towns_-LRB-film-RRB- :: sent 2):
> The film stars Nat Wolff and Cara Delevingne and was released on July 24 , 2015 , in the United States by 20th Century Fox .

**NLI probs:** contra=1.000, entail=0.000, neutral=0.000

**Verdict:** ❌ Fail

**Notes:** Both 2015 films but completely different casts.

---

### 13. uid=7728cfef  cos=0.605  contra=0.999

**Claim (SUPPORTED):** The actress who played the role of Sam Sloan's wife Trudy in the sitcom The Single Guy is English.

**Distractor** (Katherine_Bailess :: sent 0):
> Katherine Bailess -LRB- born April 24 , 1980 -RRB- is an American actress , singer , and dancer best known for playing the role of Erica Marsh on the CW 's hit show One Tree Hill , Life and Death Brigade member Stephanie on Gilmore Girls , and Kyle Hart on the VH1 series Hit the Floor .

**NLI probs:** contra=0.999, entail=0.000, neutral=0.001

**Verdict:** ❌ Fail

**Notes:** Different actress with different roles. NLI flagged on the `American actress / English actress` pattern conflict.

---

### 14. uid=f17d1edb  cos=0.624  contra=0.992

**Claim (REFUTED):** Bruce Guthro and another rock musician are not both members of the alternative rock band. The other rock musician sins on the band's song "All I need". The sound of Films of Colour was compared to the band.

**Distractor** (Monkey_on_My_Back :: sent 0):
> `` Monkey on My Back '' is a song by American hard rock band Aerosmith .

**NLI probs:** contra=0.992, entail=0.000, neutral=0.008

**Verdict:** ❌ Fail

**Notes:** Different band, different song. No overlap with Guthro or Films of Colour.

---

### 15. uid=96aa4010  cos=0.588  contra=0.985

**Claim (SUPPORTED):** The same man who fought for the precious gift that Martin Bormann gave Hitler had a dachshund.

**Distractor** (The_Ugly_Dachshund :: sent 0):
> The Ugly Dachshund is a 1966 Walt Disney Productions feature film starring Dean Jones and Suzanne Pleshette in a story about a Great Dane who believes he 's a dachshund .

**NLI probs:** contra=0.985, entail=0.000, neutral=0.015

**Verdict:** ❌ Fail

**Notes:** Single keyword overlap ("dachshund"). Completely different topic — Disney film vs WWII history.

---

### 16. uid=c23de574  cos=0.557  contra=0.522 ⚠ low-conf

**Claim (SUPPORTED):** HPC has a higher population than the birthplace of Han Feng.

**Distractor** (Palo_Alto,_California :: sent 5):
> Palo Alto was established by Leland Stanford Sr. when he founded Stanford University , following the death of his son , Leland Stanford Jr. .

**NLI probs:** contra=0.522, entail=0.001, neutral=0.477

**Verdict:** ❌ Fail

**Notes:** Random history of an unrelated city. Padded with relaxed threshold; not adversarial.

---

### 17. uid=402de9ef  cos=0.609  contra=0.999

**Claim (SUPPORTED):** The star of The Keeping Hours did not play Margo Dunne in a 2014 film set in Missouri.

**Distractor** (Playing_for_Keeps_-LRB-2012_film-RRB- :: sent 0):
> Playing for Keeps is a 2012 American romantic comedy film directed by Gabriele Muccino , starring Gerard Butler with Jessica Biel , Catherine Zeta-Jones , Dennis Quaid , Uma Thurman and Judy Greer in supporting roles .

**NLI probs:** contra=0.999, entail=0.000, neutral=0.001

**Verdict:** ❌ Fail

**Notes:** Different film entirely. Doesn't address The Keeping Hours or Margo Dunne.

---

### 18. uid=631928e7  cos=0.659  contra=0.980

**Claim (REFUTED):** The home of a race is also the venue for the FIA Formula One Canadian Grand Prix. It is home to the first formula one race won by a black driver. The race featured Formula One drivers from Canada for the first time since 2006.

**Distractor** (Argentine_Grand_Prix :: sent 1):
> Although it is no longer on the Formula One calendar , the race has a long and varied history .

**NLI probs:** contra=0.980, entail=0.000, neutral=0.020

**Verdict:** ❌ Fail

**Notes:** Argentine GP vs Canadian GP. Different country, different race.

---

### 19. uid=e58993b0  cos=0.640  contra=1.000

**Claim (SUPPORTED):** This rock group had a second studio single titled No Pussy Blues which reached number one in 1988. They and Nick Cave and the Bad Seeds, are rock bands based out of Melbourne.

**Distractor** (Pink_Floyd :: sent 0):
> Pink Floyd were an English rock band formed in London .

**NLI probs:** contra=1.000, entail=0.000, neutral=0.000

**Verdict:** ❌ Fail

**Notes:** Different band, different city (London vs Melbourne). NLI flagged on `rock band formed in X` pattern conflict.

---

### 20. uid=402de9ef  cos=0.617  contra=1.000

**Claim (SUPPORTED):** The star of The Keeping Hours did not play Margo Dunne in a 2014 film set in Missouri.

**Distractor** (Safelight_-LRB-film-RRB- :: sent 0):
> Safelight is a 2015 American drama film , written and directed by Tony Aloupis , and starring Juno Temple , Evan Peters , Kevin Alejandro , Jason Beghe , Ariel Winter , and Christine Lahti .

**NLI probs:** contra=1.000, entail=0.000, neutral=0.000

**Verdict:** ❌ Fail

**Notes:** Different 2015 film entirely. No connection to Margo Dunne / Gone Girl.

---

## Summary

| Verdict | Count | Examples |
|---|---|---|
| Pass | 2 | #1 (M&T Bank Stadium), #4 (Wendigo) |
| Fail | 18 | rest |
| **Fail rate** | **90%** | far over the 25% gate |

## Why I'm not re-mining (yet)

Raising the NLI threshold won't help — the fails consistently have contra_prob ≥ 0.99. The failure mode is *NLI-flags-different-entities-as-contradictory*, which is invariant to the threshold. The proper fix requires entity-aware filtering (extract claim entities via NER; require distractor passage to mention at least one). That's an order of magnitude of work and an open follow-up if Phase 11 robustness eval shows the current distractors don't bite.

For now: ship these distractors as **"weakly adversarial"** — they're harder than V1's cos-only baseline but don't meet the spec's strict semantic-opposite bar. The Phase 11 robustness number will be the real test.

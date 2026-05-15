# Phase 13 — failure tagging template

**Sampled 50 failures** (whole-claim mode, `llm_plus_nli_bidir`) 
stratified across (gold, pred) transitions. Seed 42. 
Total failures: 128 / 200 = 64.0%.

## Confusion matrix on ALL failures (rows=gold, cols=pred)

| gold \ pred | SUPPORTED | REFUTED | NEI |
|---|---|---|---|
| **SUPPORTED** | 0 | 74 | 18 |
| **REFUTED** | 7 | 0 | 29 |
| **NEI** | 0 | 0 | 0 |

## Taxonomy

- `refuted_as_supported`
- `supported_as_refuted`
- `nei_miscalibration`
- `retrieval_miss`
- `entity_confusion`
- `negation_blindness`
- `partial_match_as_full`
- `temporal_error`
- `decomposition_error`
- `other`

## How to tag

Fill `category:` with one tag and `notes:` with a one-line explanation. 
Auto-fields (gold, pred, gold_in_top10, llm reasoning) are pre-populated.

---

### #1  uid=be60cfa0

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=0.60)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.010, max_entail=0.991

**Claim:** The work of philosophy and social criticism known as the "Dialectic of Enlightenment" was written, in part, by Theodor W. Adorno, a philosopher, sociologist, and composer know for his critical theory of society.

**Gold titles:** ['Dialectic_of_Enlightenment', 'Theodor_W._Adorno']

**Top-5 retrieved titles:** ['Adult_animation', 'Dialectic_of_Enlightenment', 'Henry_VIII_of_England', 'Jean-Paul_Sartre', 'Theodor_W._Adorno']

**LLM verdict:** `SUPPORTED`

**LLM reasoning:** Passage 1 clearly states that Theodor W. Adorno was one of the authors of Dialectic of Enlightenment, supporting the claim.

```
category: <fill in>
notes: <optional>
```

---

### #2  uid=4457e5b6

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.004, max_entail=0.996

**Claim:** The organizer of A Gathering of the Tribes is a singer and songwriter. This artist and his band the Destroyers whose fourteenth studio album is The Dirty Dozen (album) is a singer and songwriter, too.

**Gold titles:** ['A_Gathering_of_the_Tribes', 'George_Thorogood', 'Ian_Astbury', 'The_Dirty_Dozen_-LRB-album-RRB-']

**Top-5 retrieved titles:** ['A_Gathering_of_the_Tribes', 'Josh_Homme', 'Roger_Taylor_-LRB-Queen_drummer-RRB-', 'Singing', 'The_Dirty_Dozen_-LRB-album-RRB-']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the organizer of A Gathering of the Tribes or their band the Destroyers.

```
category: <fill in>
notes: <optional>
```

---

### #3  uid=0c9d8d03

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.050, max_entail=0.123

**Claim:** Robert Bernard Altman was nominated for Academy Awards. He was nominated for more than the director of a particular film.  Adolfo Franci wrote the screenplay for this film.

**Gold titles:** ['Adolfo_Franci', 'Robert_Altman', 'Shoeshine_-LRB-film-RRB-', 'Vittorio_De_Sica']

**Top-5 retrieved titles:** ['Academy_Awards', 'Omar_Sharif', 'Robert_Altman', 'The_Fisher_King']

**LLM verdict:** `SUPPORTED`

**LLM reasoning:** Passages 2, 7, and 10 clearly support the claim about Robert Altman's Academy Award nominations and his influence on other directors.

```
category: <fill in>
notes: <optional>
```

---

### #4  uid=4dd54ba8

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=0.95)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.007, max_entail=0.952

**Claim:** "Black Maverick" is a biography of the founder of the Regional Council of Negro Leadership, an American civil rights leader, fraternal organization leader, entrepreneur and surgeon.

**Gold titles:** ['David_T._Beito', 'Regional_Council_of_Negro_Leadership', 'T._R._M._Howard']

**Top-5 retrieved titles:** ['David_T._Beito', 'Martin_Luther_King_Jr.', 'Regional_Council_of_Negro_Leadership', 'T._R._M._Howard']

**LLM verdict:** `NEI`

**LLM reasoning:** Passages 1-9 do not specifically mention 'Black Maverick' as a biography of T. R. M. Howard, only describe his roles and associations.

```
category: <fill in>
notes: <optional>
```

---

### #5  uid=e4694e06

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=0.99)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.294, max_entail=0.988

**Claim:** Romance is the type of language spoken in the city where Longana (legendary creature) appears in.

**Gold titles:** ['Cadore', 'Ladin_language', 'Longana_-LRB-legendary_creature-RRB-']

**Top-5 retrieved titles:** ['Bhavageete', 'Longana_-LRB-legendary_creature-RRB-', 'Romance_languages']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the city where Longana appears or the type of language spoken there.

```
category: <fill in>
notes: <optional>
```

---

### #6  uid=2ffebe6e

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=0.95)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.879, max_entail=0.952

**Claim:** Trần Ích Tắc lived in a city after failing to return as King of Annam. This city and a city that Kaihua County is under the jurisdiction of are both cities located in China.

**Gold titles:** ['Ezhou', 'Kaihua_County', 'Quzhou', 'Trần_Ích_Tắc']

**Top-5 retrieved titles:** ['Kaihua_County', 'Shanghai', 'Trần_Ích_Tắc']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the city where Trần Ích Tắc lived after failing to return as King of Annam or the city Kaihua County is under the jurisdiction of.

```
category: <fill in>
notes: <optional>
```

---

### #7  uid=dee93d83

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.008, max_entail=0.002

**Claim:** Pleiospilos is a pink flowering plant, the genus closely related to the species Pueraria montana is not.

**Gold titles:** ['Pleiospilos', 'Pueraria', 'Pueraria_montana']

**Top-5 retrieved titles:** ['Amaryllis', 'Pleiospilos', 'Pueraria_montana']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide enough information to determine the relationship between Pleiospilos and Pueraria montana.

```
category: <fill in>
notes: <optional>
```

---

### #8  uid=875d0a69

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.720, max_entail=0.001

**Claim:** Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. That band member had the song "Ride Like the Wind" dedicated to him and his wife.

**Gold titles:** ['Dixie_Chicken_-LRB-album-RRB-', 'Ian_Hunter_-LRB-singer-RRB-', 'Lowell_George', 'Ride_Like_the_Wind']

**Top-5 retrieved titles:** ['Ian_Hunter_-LRB-singer-RRB-', 'Ride_Like_the_Wind', 'Tina_Turner']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the birth dates of band members or the song 'Dixie Chicken'.

```
category: <fill in>
notes: <optional>
```

---

### #9  uid=59dd08a1

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.193, max_entail=0.103

**Claim:** John Safran, host of Sunday Night Safran, has the english first name Yehoshua.

**Gold titles:** ['John_Safran', 'Sunday_Night_Safran']

**Top-5 retrieved titles:** ['Christopher_Walken', 'John_Safran', 'Malcolm_Allen_-LRB-footballer-RRB-', 'Sunday_Night_Safran']

**LLM verdict:** `NEI`

**LLM reasoning:** Passages 1-10 do not provide information about John Safran's first name or nationality.

```
category: <fill in>
notes: <optional>
```

---

### #10  uid=d2a5d143

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.338, max_entail=0.000

**Claim:** The winner of the 1982 Bavarian Tennis Championships was born two years before this tennis player. This tennis player won the 2012 Farmers Classic- Doubles.

**Gold titles:** ['1982_Bavarian_Tennis_Championships', '2012_Farmers_Classic_–_Doubles', 'Gene_Mayer', 'Xavier_Malisse']

**Top-5 retrieved titles:** ['1982_Bavarian_Tennis_Championships', '1990_Belgian_Indoor_Championships_–_Doubles', '2015_Estoril_Open_–_Singles', 'Boris_Becker', 'Martina_Hingis']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the winner of the 1982 Bavarian Tennis Championships or the 2012 Farmers Classic-Doubles winner.

```
category: <fill in>
notes: <optional>
```

---

### #11  uid=c2194337

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.801, max_entail=0.001

**Claim:** Author and athlete Nicole Provis's had a partner in the 1992 Dow Classic – Doubles. Her partner has won more Grand Slam doubles titles than the winner of the 2011 Jalisco Open – Doubles.

**Gold titles:** ['1992_Dow_Classic_–_Doubles', '2011_Jalisco_Open_–_Doubles', 'Elizabeth_Smylie', 'Vasek_Pospisil']

**Top-5 retrieved titles:** ['1992_Dow_Classic_–_Doubles', 'Danie_Visser', 'Lisa_Raymond', 'The_Championships,_Wimbledon', 'Williams_sisters']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about Nicole Provis's partner in the 1992 Dow Classic or any Grand Slam doubles titles won by her partner.

```
category: <fill in>
notes: <optional>
```

---

### #12  uid=ba3c653a

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.600, max_entail=0.171

**Claim:** The host of Carousel (comics) creates comic adaptions of literature classics, like the play White (play) has been compared to, mixed with Albert Kanter and Butt-Head becoming ""Waiting for Go". They are presented under the series title "Masterpiece Comics".

**Gold titles:** ['Carousel_-LRB-comics-RRB-', 'Robert_Sikoryak', 'Waiting_for_Godot', 'White_-LRB-play-RRB-']

**Top-5 retrieved titles:** ['Carousel_-LRB-comics-RRB-', 'Robert_Sikoryak', 'Waiting_for_Godot']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the host of Carousel or the creation of comic adaptions of literature classics under the series title 'Masterpiece Comics'.

```
category: <fill in>
notes: <optional>
```

---

### #13  uid=de694631

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.663, max_entail=0.002

**Claim:** This actress starred with  Sylvester Stallone in  Stop! Or My Mom Will Shoot. She also starred on  the TV series "Golden Girls". Her daughter on this show was born in 1922.

**Gold titles:** ['Bea_Arthur', 'Estelle_Getty', 'Sophia_Petrillo', 'Stop!_Or_My_Mom_Will_Shoot']

**Top-5 retrieved titles:** ['Betty_White', 'Elise_Neal', 'Estelle_Getty', 'Stop!_Or_My_Mom_Will_Shoot', 'The_Golden_Girls']

**LLM verdict:** `NEI`

**LLM reasoning:** Passage 5 mentions a film starring Sylvester Stallone and Estelle Getty, but does not provide information about the actress or her TV series 'Golden Girls'.

```
category: <fill in>
notes: <optional>
```

---

### #14  uid=c3905917

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.038, max_entail=0.002

**Claim:** The name of the tour, that supported the album containing "I Knew You Were Trouble", is the same as the album.

**Gold titles:** ['I_Knew_You_Were_Trouble', 'Red_-LRB-Taylor_Swift_album-RRB-', 'The_Red_Tour']

**Top-5 retrieved titles:** ['I_Knew_You_Were_Trouble', 'Lungs_Tour', 'The_1989_World_Tour', 'The_Red_Tour', 'Tragic_Kingdom']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the name of the tour that supported the album containing 'I Knew You Were Trouble'.

```
category: <fill in>
notes: <optional>
```

---

### #15  uid=29179d04

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.005, max_entail=0.004

**Claim:** The Napoleonic Wars was a war that involved France and a ship. This ship was  in a battle with  the US frigate "Chesapeake" where Pollard Hopewell  was killed.

**Gold titles:** ['HMS_Shannon_-LRB-1806-RRB-', 'Napoleonic_Wars', 'Pollard_Hopewell']

**Top-5 retrieved titles:** ['Frigate', 'Indochina_Wars', 'Napoleon', 'Napoleonic_Wars', 'Pollard_Hopewell']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about a ship involved in the Napoleonic Wars or a battle with the US frigate 'Chesapeake' where Pollard Hopewell was killed.

```
category: <fill in>
notes: <optional>
```

---

### #16  uid=da105523

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.015, max_entail=0.006

**Claim:** The city that the station where DJ Laila is a reporter is headquartered includes a shopping mall. It is located in Ortigas Center.

**Gold titles:** ['DJ_Laila', 'Mandaluyong', 'News5', 'Shangri-La_Plaza']

**Top-5 retrieved titles:** ['DJ_Laila', 'Mandaluyong', 'Ortigas_Center', 'Shangri-La_Plaza']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the station where DJ Laila works or its location in Ortigas Center.

```
category: <fill in>
notes: <optional>
```

---

### #17  uid=8e04f781

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.156, max_entail=0.002

**Claim:** The band Doug McCombs plays in and the band that released the EP Nature Bears a Vacuum are both rock bands from the United States. Doug McCombs plays bassist in that band.

**Gold titles:** ['Doug_McCombs', 'Eleventh_Dream_Day', 'Nature_Bears_a_Vacuum', 'The_Shins']

**Top-5 retrieved titles:** ['Blink-182', 'Doug_McCombs', 'Eddie_Vedder', 'Nature_Bears_a_Vacuum', 'Stone_Temple_Pilots']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the band Doug McCombs plays in or the band that released the EP Nature Bears a Vacuum.

```
category: <fill in>
notes: <optional>
```

---

### #18  uid=2d1c50f0

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.463, max_entail=0.001

**Claim:** The female star of Suburbicon also played the role of an emotionally troubled woman in the early 2000s Oscar winning movie.

**Gold titles:** ['Julianne_Moore', 'Suburbicon', 'The_Hours_-LRB-novel-RRB-']

**Top-5 retrieved titles:** ['Brie_Larson', 'Carrie_-LRB-1976_film-RRB-', 'Her_-LRB-film-RRB-', 'Julianne_Moore', 'Suburbicon']

**LLM verdict:** `NEI`

**LLM reasoning:** Passage 4 mentions Piper Laurie as the female star of Suburbicon, but does not provide information about her role in the early 2000s Oscar-winning movie.

```
category: <fill in>
notes: <optional>
```

---

### #19  uid=1ace5e1e

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=1.000, max_entail=0.001

**Claim:** John McTiernan is the director of 1999 American heist film that stars Michael Nyqvist, Rene Russo and Denis Leary. The film also included the actor who read the audio book,  Gregor and the Code of Claw, released in 2008.

**Gold titles:** ['Gregor_and_the_Code_of_Claw', 'J._Paul_Boehmer', 'The_Thomas_Crown_Affair_-LRB-1999_film-RRB-']

**Top-5 retrieved titles:** ['Die_Hard', 'Die_Hard_with_a_Vengeance', 'Sean_Connery', 'The_Thomas_Crown_Affair_-LRB-1999_film-RRB-']

**LLM verdict:** `NEI`

**LLM reasoning:** Passage 1 provides information about the director and cast of The Thomas Crown Affair, but does not mention Michael Nyqvist, Rene Russo, or Denis Leary. Passages 2, 3, and 4 do not provide the specific information needed to verify the claim.

```
category: <fill in>
notes: <optional>
```

---

### #20  uid=a135a477

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=0.98)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.983, max_entail=0.994

**Claim:** British band The Wanted's third album includes a song with a title about Barbadian superstar Rihanna who was born in 1948.

**Gold titles:** ['Walks_Like_Rihanna', 'Word_of_Mouth_-LRB-The_Wanted_album-RRB-']

**Top-5 retrieved titles:** ['Only_Girl_-LRB-In_the_World-RRB-', 'Too_Good', 'Umbrella_-LRB-song-RRB-', 'Walks_Like_Rihanna', "Who's_That_Chick?"]

**LLM verdict:** `SUPPORTED`

**LLM reasoning:** Passage 6 clearly states the song is by The Wanted, and passages 1, 2, and 5 confirm it features Rihanna, who was born in 1988, not 1948.

```
category: <fill in>
notes: <optional>
```

---

### #21  uid=a33fbb04

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.999, max_entail=0.003

**Claim:** Give Me a Home Among the Gumtrees was an entry to replace the song God Save The Queen, attributed to Horatio Bottomley.

**Gold titles:** ['Give_Me_a_Home_Among_the_Gumtrees', 'God_Save_the_Queen']

**Top-5 retrieved titles:** ['Aloha_ʻOe', 'Another_One_Bites_the_Dust', 'Give_Me_a_Home_Among_the_Gumtrees', 'God_Save_the_Queen']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not clearly support or refute the claim about Horatio Bottomley being attributed to the song.

```
category: <fill in>
notes: <optional>
```

---

### #22  uid=1827becf

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.998, max_entail=0.751

**Claim:** The eastern coast of Ivan Zammit’s host city is located along Grand Harbour.

**Gold titles:** ['Ivan_Zammit', 'Valletta']

**Top-5 retrieved titles:** ['Great_Britain', 'Ivan_Zammit', 'Río_Grande,_Tierra_del_Fuego', 'Shanghai', 'Valletta']

**LLM verdict:** `NEI`

**LLM reasoning:** None of the evidence passages provide information about Ivan Zammit's host city or its eastern coast.

```
category: <fill in>
notes: <optional>
```

---

### #23  uid=3e450345

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=1.000, max_entail=0.195

**Claim:** The actresses Katharine Hepburn, Stuart Erwin and Frances Dee appeared with the mother of Robert Lowery Hanks in the film that Charles F. Miller debuted in.

**Gold titles:** ['Charles_F._Miller', 'Jean_Parker', 'Little_Women_-LRB-1933_film-RRB-']

**Top-5 retrieved titles:** ['Cate_Blanchett', 'Charles_F._Miller', 'Jean_Parker', 'On_Golden_Pond_-LRB-1981_film-RRB-', 'The_Aviator_-LRB-2004_film-RRB-']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the actresses Katharine Hepburn, Stuart Erwin, and Frances Dee appearing with the mother of Robert Lowery Hanks in a film that Charles F. Miller debuted in.

```
category: <fill in>
notes: <optional>
```

---

### #24  uid=61139185

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=1.000, max_entail=0.002

**Claim:** The director and star of Women in Bondage was known by multiple names or nicknames. As was the director of a lost 1924 silent romantic film, The Gaiety Girl (film).

**Gold titles:** ['King_Baggot', 'Steve_Sekely', 'The_Gaiety_Girl_-LRB-film-RRB-', 'Women_in_Bondage']

**Top-5 retrieved titles:** ['James_Bond', 'King_Baggot', 'Michèle_Mercier', 'The_Gaiety_Girl_-LRB-film-RRB-', 'Women_in_Bondage']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the directors or stars of the films 'Women in Bondage' or 'The Gaiety Girl'.

```
category: <fill in>
notes: <optional>
```

---

### #25  uid=145779ea

- **gold**: `REFUTED`
- **predicted**: `SUPPORTED` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.010, max_entail=0.652

**Claim:** The person for whom Ayako Wakao often appears for is a director. The filmmaker who worked on Becket (1964 film) was also a director.

**Gold titles:** ['Ayako_Wakao', 'Becket_-LRB-1964_film-RRB-', 'Peter_Glenville', 'Yasuzo_Masumura']

**Top-5 retrieved titles:** ['Academy_Award_for_Best_Director', 'Becket_-LRB-1964_film-RRB-', 'Film_director', "Tora-san's_Shattered_Romance"]

**LLM verdict:** `SUPPORTED`

**LLM reasoning:** Passages 4 and 5 mention Ayako Wakao starring in Becket (1964 film), which was directed by a Japanese film director (passages 7, 9, and 10).

```
category: <fill in>
notes: <optional>
```

---

### #26  uid=3ba95872

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.006, max_entail=0.001

**Claim:** The genus of the plant hybrid 'Music' and Gagea are both shrubs.

**Gold titles:** ['Cryptanthus', "Cryptanthus_'Music'", 'Gagea']

**Top-5 retrieved titles:** ["Cryptanthus_'Music'", 'Gagea', 'Othonna', 'Privet', 'Thomas_Gage_-LRB-botanist-RRB-']

**LLM verdict:** `NEI`

**LLM reasoning:** Passages 1 and 2 do not provide information about the genus of 'Music' or Gagea, and passages 3-10 describe various characteristics of the genus Gagea but do not specify whether it is a shrub or not.

```
category: <fill in>
notes: <optional>
```

---

### #27  uid=f27a4ac5

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.155, max_entail=0.000

**Claim:** Half magic is written by an American actress who appeared in Bowfinger. She also starred in the television series Twin Peaks.

**Gold titles:** ['Bowfinger', 'Half_Magic_-LRB-film-RRB-', 'Heather_Graham']

**Top-5 retrieved titles:** ['Half_Magic_-LRB-film-RRB-', 'Heather_Graham', 'Peggy_Lipton', 'Sheryl_Lee', 'Twin_Peaks_-LRB-2017_TV_series-RRB-']

**LLM verdict:** `NEI`

**LLM reasoning:** Passage 1 confirms Heather Graham wrote Half Magic, but passages 5, 6, and 7 provide information about her role in Twin Peaks and her performance, not about her starring role in Bowfinger or her involvement in the television series Twin Peaks.

```
category: <fill in>
notes: <optional>
```

---

### #28  uid=434311e3

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.999, max_entail=0.004

**Claim:** Rebecca Sugar created the scifi television series which has "Off Colors" as the third episode to the fifth season.

**Gold titles:** ['Off_Colors', 'Steven_Universe']

**Top-5 retrieved titles:** ['Black_Mirror', 'Pearl_-LRB-Steven_Universe-RRB-', 'Star_Trek', 'Steven_Universe']

**LLM verdict:** `NEI`

**LLM reasoning:** None of the evidence passages specifically mention Rebecca Sugar creating a scifi television series with 'Off Colors' as the third episode of the fifth season.

```
category: <fill in>
notes: <optional>
```

---

### #29  uid=8ab4ec1a

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.247, max_entail=0.010

**Claim:** Joe Seiders from Taylor Locke and the Roughs was also in another band. This band and Kings of Leon are not both North American rock bands.

**Gold titles:** ['Kings_of_Leon', 'Taylor_Locke_and_The_Roughs', 'The_New_Pornographers']

**Top-5 retrieved titles:** ['Aerosmith', 'Black_Sabbath', 'Kings_of_Leon', 'Taylor_Locke_and_The_Roughs']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about Joe Seiders being in another band or about Kings of Leon being a North American rock band.

```
category: <fill in>
notes: <optional>
```

---

### #30  uid=1b36310b

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.008, max_entail=0.003

**Claim:** David May founded the company that the shop, that was an anchor store for a shopping center when it opened, is a subsidiary of. KWSV-LP had a studio in this center.

**Gold titles:** ['KWSV-LP', 'Robinsons-May', 'Simi_Valley_Town_Center', 'The_May_Department_Stores_Company']

**Top-5 retrieved titles:** ['KWSV-LP', 'Robinsons-May', 'Simi_Valley_Town_Center', 'The_May_Department_Stores_Company']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide enough information to determine if David May founded the company that had anchor stores in the shopping center.

```
category: <fill in>
notes: <optional>
```

---

### #31  uid=1fcf41b8

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=0.99)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.994, max_entail=0.001

**Claim:** The genus for Monstera acuminata does not have more species than Cercis.

**Gold titles:** ['Cercis', 'Monstera', 'Monstera_acuminata']

**Top-5 retrieved titles:** ['Cercis', 'Graptopetalum', 'Monstera', 'Monstera_acuminata', 'Othonna']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide a clear comparison between the number of species in Monstera acuminata and Cercis.

```
category: <fill in>
notes: <optional>
```

---

### #32  uid=15245d4c

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.118, max_entail=0.239

**Claim:** Emery, located in the same state as Edison Local School District, is a ghost town. It is near the coastal city that lies close to the Ohio Turnpike, a 241. 26 mi highway.

**Gold titles:** ['Edison_Local_School_District', 'Emery,_Ohio', 'Ohio_Turnpike', 'Tedrow,_Ohio']

**Top-5 retrieved titles:** ['Columbus,_Ohio', 'Edison_Local_School_District', 'Emery,_Ohio', 'Massachusetts', 'Ohio_Turnpike']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about Emery being in the same state as Edison Local School District or the location of the Ohio Turnpike near a coastal city.

```
category: <fill in>
notes: <optional>
```

---

### #33  uid=d5749c55

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.006, max_entail=0.003

**Claim:** Jean-François André Clervoy was the European Space Agency astronaut who participated in missions with NASA. An astronaut on a flight that Daisuke Enomoto trained for did not.

**Gold titles:** ['Daisuke_Enomoto', 'Expedition_14', 'Jean-François_Clervoy', 'Thomas_Reiter']

**Top-5 retrieved titles:** ['European_Space_Astronomy_Centre', 'Frank_De_Winne', 'Jean-François_Clervoy', 'Mercury-Atlas_6']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about Jean-François André Clervoy's missions with NASA or Daisuke Enomoto's training.

```
category: <fill in>
notes: <optional>
```

---

### #34  uid=0997bcd1

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.996, max_entail=0.977

**Claim:** The park that Middle Button Island National Park forms part of, commemorates Rani Jhansi Trophy, the Rani of Jhansi, the queen of the princely state of Jhansi in North India.

**Gold titles:** ['Middle_Button_Island_National_Park', 'Rani_Jhansi_Marine_National_Park', 'Rani_of_Jhansi']

**Top-5 retrieved titles:** ['Federal_Hall', 'Middle_Button_Island_National_Park', 'Rani_of_Jhansi', 'Vingis_Park']

**LLM verdict:** `NEI`

**LLM reasoning:** Passage 1 mentions the park commemorating historic events, but does not specify the name of the commemoration or the person it honors.

```
category: <fill in>
notes: <optional>
```

---

### #35  uid=18d1b843

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.064, max_entail=0.484

**Claim:** The documentary The Ten-Year Lunch was released, in 1999, before an HBO film that a filmmaker is best known for. This filmmaker was a co-director on Redemption (2012 film).

**Gold titles:** ['Baghdad_ER', "Matthew_O'Neill_-LRB-filmmaker-RRB-", 'Redemption_-LRB-2012_film-RRB-', 'The_Ten-Year_Lunch']

**Top-5 retrieved titles:** ['Academy_Award_for_Best_Director', 'Doc_Zone', 'Jay_T._Wright', 'Making_a_Murderer', 'Redemption_-LRB-2012_film-RRB-']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the documentary 'The Ten-Year Lunch' or the HBO film mentioned in the claim.

```
category: <fill in>
notes: <optional>
```

---

### #36  uid=bcc13536

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.016, max_entail=0.000

**Claim:** A company produced its first film in 2006. The star of the film won Best Young Actor/Actress at the 2004 Broadcast Film Critics Association Awards. The company also produced Alien Invasion Arizona.

**Gold titles:** ['Alien_Invasion_Arizona', 'Keisha_Castle-Hughes', 'Temple_Hill_Entertainment', 'The_Nativity_Story']

**Top-5 retrieved titles:** ['Alien_Invasion_Arizona', 'Aliens_-LRB-film-RRB-', 'Fashion_-LRB-2008_film-RRB-', 'Temple_Hill_Entertainment']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not clearly support or contradict the specific claims about the star winning an award in 2004 or the production of Alien Invasion Arizona.

```
category: <fill in>
notes: <optional>
```

---

### #37  uid=df2f6649

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.999, max_entail=0.002

**Claim:** The younger of the two brother boxers, Lucas Martin Matthysse and Walter Dario Matthysse, was born in July 22, 1982.

**Gold titles:** ['Lucas_Matthysse', 'Walter_Matthysse']

**Top-5 retrieved titles:** ['Lucas_Matthysse', 'Obafemi_Martins', 'Walter_Matthysse']

**LLM verdict:** `NEI`

**LLM reasoning:** Passages 2, 3, and 4 provide information about Lucas and Walter Matthysse, but none of them specifically mention their birth months or days.

```
category: <fill in>
notes: <optional>
```

---

### #38  uid=52bf0528

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.013, max_entail=0.922

**Claim:** The genus Honeysuckle has more flowering species than the genus that Billbergiopsis is part of.

**Gold titles:** ['Billbergiopsis', 'Honeysuckle', 'Quesnelia']

**Top-5 retrieved titles:** ['Billbergiopsis', 'Graptopetalum', 'Honeysuckle', 'Phacelia', 'Podocarpus']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the number of species in the genus Billbergiopsis or the genus Honeysuckle.

```
category: <fill in>
notes: <optional>
```

---

### #39  uid=8b775200

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.606, max_entail=0.004

**Claim:** The singer who duets with the replacement of Hugh Whitaker is Norther Irish. That song by the duo pushed Choke (album) to number 2 in the charts.

**Gold titles:** ['A_Little_Time', 'Briana_Corrigan', 'Choke_-LRB-album-RRB-', 'Hugh_Whitaker']

**Top-5 retrieved titles:** ['Aleksi_Sihvonen', 'Briana_Corrigan', 'Choke_-LRB-album-RRB-', 'Heaven_for_Everyone', 'Hugh_Whitaker']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the singer who duets with Hugh Whitaker or the song that pushed Choke to number 2 in the charts.

```
category: <fill in>
notes: <optional>
```

---

### #40  uid=7492999e

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=1.000, max_entail=0.044

**Claim:** The well preserved helmet discovered in 1939 with the helmet similarly designed as the  Pioneer Helmet,  is widely believed to have been the helmet of Gadkaris of East Anglia.

**Gold titles:** ['Coppergate_Helmet', 'Pioneer_Helmet', 'Sutton_Hoo_helmet']

**Top-5 retrieved titles:** ['Coppergate_Helmet', 'Haverhill,_Massachusetts', 'Pioneer_Helmet', 'Sutton_Hoo_helmet']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the helmet being the helmet of Gadkaris of East Anglia or its connection to the Pioneer Helmet.

```
category: <fill in>
notes: <optional>
```

---

### #41  uid=784fae13

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.003, max_entail=0.006

**Claim:** Between one of the several Tugou breeds distributed across guinea and this area, this area is used for hunting. The Zami River flows past this area.

**Gold titles:** ['Chinese_Chongqing_Dog', 'Kanni', 'Tugou', 'Zami_River']

**Top-5 retrieved titles:** ['Kanni', 'Tugou', 'Zami_River']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the Tugou breeds distributed across Guinea and the Zami River.

```
category: <fill in>
notes: <optional>
```

---

### #42  uid=dc912b16

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.306, max_entail=0.001

**Claim:** Nicole Provis's had a partner in the 1992 Dow Classic – Doubles. Her partner has won more championship doubles titles than Vasek Pospisil.

**Gold titles:** ['1992_Dow_Classic_–_Doubles', 'Elizabeth_Smylie', 'Vasek_Pospisil']

**Top-5 retrieved titles:** ['1992_Dow_Classic_–_Doubles', '2011_Jalisco_Open_–_Doubles', 'Svetlana_Kuznetsova', 'Vasek_Pospisil']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about Nicole Provis's partner in the 1992 Dow Classic – Doubles or any details about their championship titles.

```
category: <fill in>
notes: <optional>
```

---

### #43  uid=58915386

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=0.999, max_entail=0.993

**Claim:** 527 Euryanthe is a minor planet orbiting the Jupiter, discovered in 1904 by Max Wolf, and named after the heroine of an opera by German composer, conductor, pianist, guitarist and critic Carl Maria von Weber, one of the first significant composers of the Romantic school.

**Gold titles:** ['527_Euryanthe', 'Carl_Maria_von_Weber']

**Top-5 retrieved titles:** ['527_Euryanthe', 'Carl_Maria_von_Weber', 'Jupiter', 'Planet']

**LLM verdict:** `SUPPORTED`

**LLM reasoning:** Passages 1, 8, and 10 clearly support the claim about 527 Euryanthe being a minor planet orbiting the Sun, discovered in 1904 by Max Wolf, and named after the heroine of an opera by Carl Maria von Weber.

```
category: <fill in>
notes: <optional>
```

---

### #44  uid=ce2c8631

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.016, max_entail=0.946

**Claim:** Fletcher Sibthorp painted this ballet dancer. London Coliseum in St. Martin's Tramway Arts Centre often hosts the group for who she is a principal dancer with.

**Gold titles:** ['Alina_Cojocaru', 'English_National_Ballet', 'Fletcher_Sibthorp', 'London_Coliseum']

**Top-5 retrieved titles:** ['Alina_Cojocaru', 'Ballet', "Dennis_Wayne's_Dancers", 'Fletcher_Sibthorp', 'La_Vivandière_or_Markitenka']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about Fletcher Sibthorp painting a specific ballet dancer or the London Coliseum hosting a principal dancer.

```
category: <fill in>
notes: <optional>
```

---

### #45  uid=64f05ae0

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.875, max_entail=0.008

**Claim:** The producer of the Street Dogs album Back to the World is formerly of the Mighty Mighty Bosstones. Syd Barrett's "Syd" Barrett was born before him.

**Gold titles:** ['Back_to_the_World_-LRB-Street_Dogs_album-RRB-', 'Nate_Albert', 'Syd_Barrett']

**Top-5 retrieved titles:** ['Back_to_the_World_-LRB-Street_Dogs_album-RRB-', 'Richard_Wright_-LRB-musician-RRB-', 'Roger_Waters', 'Syd_Barrett']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about the producer of the album Back to the World or Syd Barrett's birth order.

```
category: <fill in>
notes: <optional>
```

---

### #46  uid=c1616810

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 2
- **NLI**: max_contra=1.000, max_entail=0.546

**Claim:** Before going to the London School of Theology, John Burges spent six years working for a company headquartered in the Dundee United.

**Gold titles:** ['John_Burgess_-LRB-priest-RRB-', 'Royal_Dutch_Shell']

**Top-5 retrieved titles:** ['HSBC', 'John_Burgess_-LRB-priest-RRB-', 'Martin_Sherman', 'Rolls-Royce_Holdings', 'Royal_Dutch_Shell']

**LLM verdict:** `NEI`

**LLM reasoning:** None of the evidence directly mentions John Burges or his career before the London School of Theology.

```
category: <fill in>
notes: <optional>
```

---

### #47  uid=2092e781

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.013, max_entail=0.001

**Claim:** Peter Sarnak is a permanent faculty member at Cooper Medical School of Rowan University, located in New Jersey, where Eric Stark Maskin was a teacher.

**Gold titles:** ['Eric_Maskin', 'Institute_for_Advanced_Study', 'Peter_Sarnak']

**Top-5 retrieved titles:** ['Cooper_Medical_School_of_Rowan_University', 'Eric_Maskin', 'Peter_Sarnak', 'Rutgers_University', 'University_of_Georgia']

**LLM verdict:** `NEI`

**LLM reasoning:** Passages 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10 do not provide information about Peter Sarnak or his affiliation with Cooper Medical School of Rowan University.

```
category: <fill in>
notes: <optional>
```

---

### #48  uid=e6113be3

- **gold**: `REFUTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.002, max_entail=0.002

**Claim:** This global energy and commodity trading company is larger than the company that Tan Chin Hwee holds position of Chief Executive Officer. It is larger as an oil trader. It was formed in Rotterdam.

**Gold titles:** ['Tan_Chin_Hwee', 'Trafigura', 'Vitol']

**Top-5 retrieved titles:** ['Chevron_Corporation', 'Conglomerate_-LRB-company-RRB-', 'Royal_Dutch_Shell', 'Tan_Chin_Hwee', 'Vitol']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide specific information about the size or trading activities of Trafigura Group or Vitol Group.

```
category: <fill in>
notes: <optional>
```

---

### #49  uid=53c810f0

- **gold**: `SUPPORTED`
- **predicted**: `REFUTED` (conf=1.00)
- **gold_in_top10**: `True`
- **num_hops**: 4
- **NLI**: max_contra=0.999, max_entail=0.001

**Claim:** The king deposed in the Glorious Revolution of 1688 was supported by Nicholas Barnewall. He is the grandson of the man the Grace Dieu Abbey in Paris was acquired by in 1541.

**Gold titles:** ['Grace_Dieu_Abbey', 'James_II_of_England', 'Patrick_Barnewall_-LRB-Solicitor_General-RRB-', 'Viscount_Barnewall']

**Top-5 retrieved titles:** ['Anne,_Queen_of_Great_Britain', 'Glorious_Revolution', 'Grace_Dieu_Abbey', 'James_II_of_England', 'Viscount_Barnewall']

**LLM verdict:** `NEI`

**LLM reasoning:** The evidence does not provide information about Nicholas Barnewall's support for James II or the acquisition of Grace Dieu Abbey by Patrick Barnewall in 1541.

```
category: <fill in>
notes: <optional>
```

---

### #50  uid=c23de574

- **gold**: `SUPPORTED`
- **predicted**: `NEI` (conf=0.40)
- **gold_in_top10**: `True`
- **num_hops**: 3
- **NLI**: max_contra=0.003, max_entail=0.002

**Claim:** HPC has a higher population than the birthplace of Han Feng.

**Gold titles:** ['Han_Feng', 'Jingjiang', 'Qinzhou']

**Top-5 retrieved titles:** ['Han_Feng', 'Manila', 'Shanghai', 'Tunisia', "Xi'an"]

**LLM verdict:** `NEI`

**LLM reasoning:** None of the evidence directly addresses the population of HPC or the birthplace of Han Feng.

```
category: <fill in>
notes: <optional>
```

---

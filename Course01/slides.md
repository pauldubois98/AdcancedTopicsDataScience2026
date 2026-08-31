---
title: |
  Session 1\
  Course Introduction & Healthcare Data
subtitle: "Advanced Data Science · NYU Paris · 2026"
author: "Paul Dubois"
date: "Week 1 — Lecture (2.5h)"
---

# 0. Welcome

## Who I am, who you are

- **Instructor:** Paul Dubois — data science & AI applied to healthcare
- **You:** 4th year undergraduates, with background in
  - linear algebra
  - probability & statistics
  - Python programming
- Around the room: name, major, one dataset you have already worked with

## What this course is

> Advanced methods in data science and applied machine learning, with case
> studies drawn from healthcare and biomedical research.

- The **theory is domain-general:**

  it transfers to finance, industry, climate, anything

- The **applications are clinical:**

  because that is my expertise

- All labs use **public datasets**

  (for legal issues)

## What this course is *not*

- Not a medical course
- Not a "call `.fit()`" course
- Not a deep learning theory course

  we will use deep learning if and only if it is appropriate

::: notes
Set expectations early: a model with 0.95 AUC that is useless in the clinic is a
failed project in this course.
:::

## Format

| | |
|---|---|
| **Lecture** | 2.5h per week: concepts, methods |
| **Recitation** | 1.5h per week: hands-on lab |
| **Duration** | 14.5 weeks |
| **Language** | English |

## Grading

| Component | Weight |
|---|---|
| Weekly labs | 0% |
| Kaggle challenges | ? % |
| Paper reading presentation/report | ? % |
| Poster mini-project | ? % |
| Final project presentation/report | ? % |

## Roadmap of the semester

- **Weeks 1–4:** Foundations (data, features, missingness...)
- **Weeks 5–6:** Time series
- **Weeks 7–10:** Images (2D and 3D)
- **Weeks 11–12:** Text (NLP/LLMs)
- **Weeks 13–14:** Causality, bias, privacy, deployment

*This syllabus is provisional and may be adjusted weekly depending on the pace of the class.*

## Tooling

- **Python 3.11+**
- `uv` for environments (or `pip`)
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- `pytorch` for deep learning
- Jupyter notebooks for labs
- Compute: a modern laptop should be enough

    if more compute is needed, we will make arrangements

# 1. What makes data science hard

## The textbook picture

> **D** = {(x⁽ⁱ⁾, y⁽ⁱ⁾)}, i = 1 … n     drawn **i.i.d.** from P(X, Y)

Given **D**, generalise to new draws from P.

- **Noise/errors:** the labels and the features are wrong, sometimes badly
- **Missingness:** the data is (sometimes) absent
- **Class imbalance:** the event you care about is rare

## Noise

Your label is not the truth: It is a **proxy** produced by the measurement process.

Example:

- "Pneumonia" on a chest X-ray: one radiologist's opinion, alone
- Inter-rater agreement between radiologists is often 40% to 70%

**Practical rule:** before modelling, ask *who* produced this label, *when*, and *why*.

Label noise caps your achievable performance: you cannot beat the annotator.

Symmetric noise hurts *variance*; asymmetric noise creates a *bias*.

::: notes
Ask: if two experts only agree 70% of the time, what does 95% accuracy mean?
:::

## Noise

![](img/noise_q.png)

::: notes
Hands up before you click on. The annotator over-calls: 35% of the true negatives
came back as findings. Ask for a show of hands - does the fitted line move left,
move right, or stay put? Take a vote, then reveal. Most rooms say "stay put"
because they are thinking of symmetric noise averaging out.
:::

## Noise

![](img/noise.png)

::: notes
Left: the labels you wish you had. Right: the same 600 points, but the annotator is
cautious and calls 35% of the negatives positive. The grey line is the truth, the
amber line is what least squares gives you. Ask the room: which way did it move,
and why is that worse than random flipping?
:::

## Noise

![](img/noise_alt.png)

::: notes
Product-review sentiment, labelled twice by crowdworkers.
Same mechanism, no radiologist in sight. Ask the room what accuracy they would put
in a paper if worker B is the test set: the honest answer is that 83% is the
ceiling, and a model reported at 90% is measuring the annotator, not the language.
Show of hands before you reveal: does a bigger model fix this? It does not - the
only fix is more annotators per item, or a better annotation guideline.
:::

## Errors in features

- Sensor artefacts: a heart rate of 300 bpm is a detached lead, not tachycardia
- Unit chaos: weight in kg *and* lb in the same column; glucose in mg/dL *and* mmol/L
- Copy-forward: the same "current" value duplicated across days of notes
- Timestamps: charted at the time of *entry*, not the time of *measurement*

## Errors in features

```python
>>> vitals.heart_rate.agg(["min", "max"])
min      0.0
max    300.0    # a detached lead

>>> vitals.weight.plot.hist(bins=60)
                # two humps: kg and lb

>>> labs.groupby("unit").glucose.median()
mg/dL    126.0
mmol/L     7.0  # same test, same patient
```

::: notes
Run these three before anything else. `min`/`max` catch impossible values, a histogram
catches mixed units far faster than reading a data dictionary, and grouping by unit
catches the column that was silently concatenated from two sites.
:::

## Errors in features

![](img/errors_alt.png)

::: notes
A delivery fleet's telemetry, checked with the same three lines.
Left: "no GPS fix" was written to the database as the number (0, 0) - a sentinel,
not a null, and it is off the coast of Africa. Exactly the same bug as a heart rate
of 0. Right: one supplier never converted mph to km/h, so a single column holds two
units and the histogram has two humps.
The expensive version of this is the Mars Climate Orbiter, lost in 1999 because one
team worked in pound-force seconds and the other in newton-seconds.
:::

## Errors in features

![](img/errors_alt_bis.png)

::: notes
The same fleet, split by supplier. Both are driving the same roads at the same
speeds - but supplier B's firmware reports mph and nobody converted it, so the
column holds 24 where it should hold 38.
Walk the panels left to right and ask after each of the first two whether anything
is wrong. Nothing is: each supplier's distribution is a clean bell curve with a
plausible mean, and every per-source sanity check passes. The bug exists only in the
concatenation, which is the one view nobody plots.
The expensive version is the Mars Climate Orbiter, lost in 1999 because one team
worked in pound-force seconds and the other in newton-seconds.
:::

## Missingness

| Mechanism | Meaning | Safe to drop? |
|---|---|---|
| **MCAR** | Missing completely at random | Yes (lose power only) |
| **MAR** | Missing depends on *observed* data | With correct modelling |
| **MNAR** | Missing depends on the *unobserved* value | No — bias is unavoidable |

- Missingness is almost never random:
- A test is measured **because** the clinician suspects a problem
- A patient with no lab results is usually very healthy
- The *absence* of a test can predict the outcome better than the test's value

::: notes
We return to this in depth in Week 2 — imputation strategies and multiple imputation.
:::

## Missingness

![](img/missingness_q.png)

::: notes
Show the matrix, let them read it for fifteen seconds, then take the vote: who died
more often, the patients whose lactate was drawn or the ones where it never was?
Hands for each. The room usually splits, which is the point - nobody has looked at
a single lactate value yet.
:::

## Missingness

![](img/missingness.png)

::: notes
Left, patients sorted by how unwell they are: routine observations are almost always
there, the specialised ones only for the sick. Right: the patients who had a lactate
drawn died five times more often — and that is before anyone looked at the number.
Impute it away and you delete the strongest signal in the table.
:::


## Guess the base rate

Write your guess down first. No phones, no neighbours.

- In-hospital mortality, ICU admissions: **\_\_\_ %**
- 30-day readmission after discharge: **\_\_\_ %**
- A rare disease at population screening: **1 in \_\_\_**

::: notes
Make them commit to a number before you say anything - written down, not shouted
out, otherwise the first loud answer anchors the room.
Rooms routinely guess far too low for ICU mortality and far too high for the
rare disease. Ask why the guess is high: because the cases you hear about are the
ones that went wrong, which is the same selection effect that shapes the data.
:::

## Guess the base rate

Write your guess down first. No phones, no neighbours.

- In-hospital mortality, ICU admissions: **~20 %**
- 30-day readmission after discharge: **~15 %**
- A rare disease at population screening: **1 in 10 000**

## Class imbalance

- Rare disease screening is 1 in 10 000

=> A model that predicts "no" always is 99.99% accurate

- Blood types are not equally spread on the population

The imbalance is changing according to the world regions

=> Can lead to distribution shift

## Class imbalance

![](img/imbalance_q.png)

::: notes
"Hands up if you would ship it." Wait for the hands, then ask one person who put
their hand up to say why, and one who did not. Then reveal. Do not rush this one -
the discomfort is the lesson.
:::

## Class imbalance

![](img/imbalance.png)

::: notes
This model has no parameters — it is `return 0`. It beats most first attempts on
accuracy and it would kill everyone it was meant to save. Every metric we use from
Week 3 onwards exists to make this model score zero.
:::

## Class imbalance

![](img/imbalance_alt.png)

::: notes
Card fraud: 492 in 284 807 transactions.
Real figures, from the ULB/Worldline dataset that half of them will meet on Kaggle.
`return 0` scores 99.83% and catches nothing. Ask what the bank actually cares
about, and you get the answer that accuracy was never the objective: they care about
euros recovered per false alarm, because every false alarm blocks a real customer's
card in a supermarket queue.
:::

## Distribution shift

Your model is trained on one slice of the world, then deployed in another:

- **Covariate shift:** P(X) changes: new hospital, new scanner, new population
- **Label shift:** P(Y) changes: a new variant, a new season
- **Concept drift:** P(Y | X) changes: treatment guidelines are updated

## Distribution shift

![](img/shift.png)

::: notes
Nothing was retrained, no code changed. The deployment hospital simply admits older
patients, and the case mix kept drifting. Without a monitoring dashboard you find out
from a complaint, not from a metric. Week 14.
:::

## Distribution shift

![](img/shift_alt.png)

::: notes
Weekly demand forecasting, and a competitor opens in week 70.
The hospital example on the previous slide was covariate shift - P(X) moved. This one
is a concept change: the relationship between the calendar and demand is simply not
the same after week 70, and no amount of retraining on pre-70 data helps.
Forecast error goes from 2% to 60% in a single week, and nobody touched the code.
Ask them how they would have found out. The honest answer, in most companies, is
"someone in finance noticed the numbers were wrong".
:::

## Shortcuts

Models learn whatever correlates with y in *the* data.

- Pneumonia detectors that learned the **portable X-ray marker** (sicker patients get
  bedside imaging)
- Skin lesion classifiers that learned the **surgical ruler** placed next to malignant lesions
- Portrait to age models looking at **ears size**

::: notes
Zech et al. 2018 (cross-site CNN generalisation); Esteva/ISIC ruler artefact.
These come back in Week 7.
:::

## Shortcuts: what the model actually learned

![](img/shortcut_q.png)

::: notes
Two training images and a perfect score. Give them thirty seconds in pairs, then
take two or three guesses out loud before revealing the third panel. Someone
usually spots the marker; if nobody does, that is the more interesting outcome.
:::

## Shortcuts: what the model actually learned

![](img/shortcut.png)

::: notes
Every "pneumonia" scan in this hospital was shot at the bedside, so the portable
unit's marker is in the corner of every positive. The model finds it in one epoch,
scores perfectly, and collapses the moment you deploy it anywhere else.
The marker here is drawn on — the phenomenon is Zech et al. 2018.
:::

## Shortcuts

![](img/shortcut_alt.png)

::: notes
Huskies and wolves — the classifier that learned snow.
Ribeiro et al. 2016, the LIME paper. They trained a deliberately bad husky-vs-wolf
classifier, and the explanation showed it was scoring the background: every wolf
photo had snow in it. When they showed the predictions to graduate students, most
trusted the model - until they saw where it was looking.
The images here are drawn, not photographs; the experiment is real.
:::

## Leakage

Information at training time that will not exist at prediction time.

Typically:

- The same patient in both train and test (e.g. duplicated)
- Feature that is a deterministic function of the outcome
  (e.g. `discharge_location == 'DIED'`)
- Bug in the train/test split

## Spot the leakage

```python
# A
X = StandardScaler().fit_transform(X)
tr, te = train_test_split(X)

# B
df["los"] = df.discharge - df.admit
model.fit(df[["los", "age"]], df.died)

# C
ids = df.subject_id.unique()
tr, te = train_test_split(ids)

# D
df = df.sort_values("charttime")
tr, te = df[:8000], df[8000:]
```

Which of these leak?

::: notes
Vote by fingers on each one before discussing. A leaks: the scaler saw the test set.
B leaks: length of stay is only known at discharge, and death shortens it. C is the
right idea - split on patients, not rows. D is the one worth arguing about: a
time-ordered split is good practice against shift, but the same patient can still
straddle the cut, so it leaks by group unless you also split by patient.
Answer: A, B and D leak. Only C is safe.
:::

## A — leaks: the scaler saw everything

```python
# WRONG
X = StandardScaler().fit_transform(X)
tr, te = train_test_split(X)

# RIGHT
tr, te = train_test_split(X)
scaler = StandardScaler().fit(X[tr])
X_tr = scaler.transform(X[tr])
X_te = scaler.transform(X[te])
```

The mean and variance were computed over the test rows too.

::: notes
The optimism is small but real, and it grows fast as p grows or n shrinks. The same
bug hides inside feature selection, PCA, and target encoding - anything fitted before
the split. The durable fix is a `Pipeline`, so the preprocessing is refitted inside
every CV fold rather than remembered from the whole dataset.
:::

## B — leaks: the feature knows the answer

```python
# WRONG
df["los"] = df.discharge - df.admit
model.fit(df[["los", "age"]], df.died)

# RIGHT - freeze features at a cut time
cut = df.admit + pd.Timedelta(hours=24)
feats = events[events.time <= cut]
model.fit(summarise(feats), df.died)
```

Length of stay is only known once the stay is over — and dying ends it.

::: notes
This is target leakage, and it is the one that produces AUCs of 0.99. Ask the room
what a deployed model would actually have at prediction time: a patient who is still
in a bed has no discharge date. The general rule is to pick a decision time and
refuse every value recorded after it.
:::

## C — safe: the split is on patients

```python
ids = df.subject_id.unique()
tr, te = train_test_split(ids)
```

One patient's rows now land entirely on one side.

::: notes
This is the right idea. The version worth writing is `GroupShuffleSplit`, which does
the same thing without you handling ids by hand, and which composes with
cross-validation. Note it is necessary but not sufficient - C is still vulnerable to
the time problem in D if the model will be deployed forwards in time.
:::

## D — leaks: same patient, both sides

```python
# WRONG - a stay can straddle the cut
df = df.sort_values("charttime")
tr, te = df[:8000], df[8000:]

# RIGHT - split by time AND by patient
cut = df.charttime.quantile(0.8)
ids = df[df.charttime <= cut].subject_id
te = df[~df.subject_id.isin(ids)]
```

A time split is the right instinct; on its own it still leaks by group.

::: notes
This is the one worth arguing about, and the honest answer is "both". Splitting by
time is what you want if the model will run forwards - it is the only split that
measures shift. But a stay spanning the cut puts the same patient in both halves.
You need both constraints, and the two fight each other: patients admitted before
the cut and discharged after it have to be thrown away.
:::


## Autopsy

Name the failure mode for each:

1. **Google Flu Trends:** tracked flu well for years, then overshot the 2013 peak
   by roughly double, and was retired
2. **Epic's sepsis model:** sold on an AUC of 0.76–0.83, scored 0.63 in an
   independent evaluation of ~28 000 admissions
3. **COVID chest X-ray models:** hundreds published in 2020, an independent review
   found none fit for clinical use

::: notes
Give them three minutes in pairs, collect answers on the board under the four
headings, then reveal the next slide.
Answers, and none is a single cause:
1. Concept drift, plus a feedback loop - Google's own autocomplete changed what
   people searched for, so the features moved under the model.
2. Shift and label noise: sepsis is defined by a billing code that varies by site,
   and the model was tuned somewhere else. Wong et al., JAMA Intern Med 2021.
3. Shortcuts, mostly - models learned the source hospital, scanner and laterality
   tokens rather than the disease. Roberts et al., Nat Mach Intell 2021;
   DeGrave et al. 2021.
Collect answers on the board under the four headings before moving on.
:::

## Autopsy

Name the failure mode for each:

1. Google Flu Trends: **Concept drift**, plus a *feedback loop*
2. Epic's sepsis model: **Shift** + *label noise* in the outcome
3. COVID chest X-ray models: **Shortcuts**, and **leakage** across sources

::: notes
Insist that none of the three has a single cause - that is why the four headings
are a checklist and not a taxonomy.
Flu Trends: the search terms drifted, partly because Google's own autocomplete
changed what people typed, so the features moved under a model nobody retrained.
Epic: sepsis is defined by a billing code applied differently at every site, so
both P(X) and the meaning of Y moved. Wong et al., JAMA Intern Med 2021.
COVID: models learned the source hospital, the scanner and laterality tokens.
Many also pooled adult pneumonia with paediatric controls, so the classifier had
only to detect a child. Roberts et al., Nat Mach Intell 2021; DeGrave et al. 2021.
:::

## Break time

![](img/break.png)

::: notes
Write the resume time on the board before you walk off. "Ten minutes" announced
from the front of a lecture theatre reliably becomes fifteen, and this is the
half-way point of a 2.5h session - section 2 still has to fit.
This is also the moment the students who will not put a hand up in front of the
room will come and ask you something, so stay near the front for the first few
minutes.
:::

# 2. Data types and example

## Tabular: one row per unit `n × p`

:::::: columns
::: column
**Ward interventions**

![](img/tabular_medical.jpg)
:::
::: column
**A ship's passenger manifest**

![](img/tabular_other.jpg)
:::
::::::

::: notes
The workhorse: gradient boosting still wins here more often than deep learning.
The hard part is never the model, it is deciding what one row *is*.
Left: Cortejoso et al., Clin Interv Aging (CC BY 3.0).
Right: US Bureau of Immigration, 1912 (CC0) — the Titanic dataset you have all seen.
:::

## 1D signals: a value over time `n × T`

:::::: columns
::: column
**A 12-lead ECG**

![](img/signal_medical.jpg)
:::
::: column
**An audio waveform**

![](img/signal_other.jpg)
:::
::::::

::: notes
Same object: a regularly sampled sequence. Same toolbox: filtering, spectrograms,
1D convolutions, transformers. The ECG is 500 Hz, the audio 44.1 kHz — three orders
of magnitude apart, identical methods.
Left: Glenlarson (public domain). Right: Em3rgent0rdr (CC BY-SA 4.0).
:::

## 2D images: a grid of pixels `H × W × C`

:::::: columns
::: column
**A chest radiograph**

![](img/image2d_medical.jpg)
:::
::: column
**A satellite view of farmland**

![](img/image2d_other.jpg)
:::
::::::

::: notes
Both are arrays of numbers on a grid, and a CNN or a ViT does not know the difference.
What differs is the *label*: "pneumonia" is one radiologist's opinion, "wheat" is
ground truth you can go and check.
Left: Mikael Häggström (CC0). Right: NASA/ASTER, Kansas (public domain).
:::

## 3D volumes: a stack of slices, `D × H × W`

:::::: columns
::: column
**A CT**

![](img/volume3d_medical.jpg)

*(three planes view)*
:::
::: column
**A LiDAR point cloud**

![](img/volume3d_other.jpg)
:::
::::::

::: notes
The third dimension costs you: memory grows cubically, labelled data gets scarcer,
and pretrained weights are far rarer than in 2D. Note the two are not the same kind
of 3D — the CT is a dense voxel grid, the point cloud is sparse and unordered.
Left: Mikael Häggström (CC0). Right: Daniel L. Lu, San Francisco (CC BY 4.0).
:::

## Text: a sequence of tokens

:::::: columns
::: column
**A radiology report**

![](img/text_medical.jpg)
:::
::: column
**A newspaper page**

![](img/text_other.jpg)
:::
::::::

::: notes
Clinical text is its own dialect: abbreviations, negation ("no evidence of
hydrocephalus"), copy-paste, and dictation errors. A model trained on newspapers
reads "PT" as a person and not as prothrombin time. Weeks 11–12.
Left: Tu et al., MultiMedBench (CC BY 4.0). Right: The Echo, 1920 (public domain).
:::

# 3. Where data science is used

## The same five datatypes

![](img/dsmap.png)

::: notes
The columns are section 2's five data types. The point is the density: no column
belongs to one field, and no field lives in one column.
Ask which cell surprises them. Usual answer is 3D volumes for sport - it is pose and
player tracking reconstructed in three dimensions, and it is the same convolution.
:::

## Finance

:::::: columns
::: column
**Fraud detection at a card network**

- **In:** one transaction — amount, merchant, time, this card's history
- **Out:** P(fraud), in under 100 ms
- **Fails on:** imbalance — 0.2% positive, and an adversary who adapts
:::
::: column
![](img/task_finance.png)
:::
::::::

::: notes
The card-fraud figure from section 1 was this. Worth adding the adversarial twist:
unlike a disease, fraud pushes back. Retrain monthly or the pattern you learned is
the one the fraudsters have already abandoned.
The 100 ms is not decoration - it rules out most of what we will cover in week 3.
:::

## Retail

:::::: columns
::: column
**Demand forecasting for a supermarket chain**

- **In:** one product in one store — two years of weekly sales, price, promotions
- **Out:** units sold, each of the next 14 weeks
- **Fails on:** distribution shift — promotions, weather, a competitor
:::
::: column
![](img/task_retail.png)
:::
::::::

::: notes
The demand-forecast figure in section 1 was this one. Millions of series, most of
them mostly zero: intermittent demand is its own literature, and it is imbalance
wearing a time series costume.
:::

## Manufacturing

:::::: columns
::: column
**Visual inspection on a production line**

- **In:** one part, photographed as it passes
- **Out:** defect or not, and where on the part
- **Fails on:** imbalance *and* label noise, together
:::
::: column
![](img/task_manufacturing.png)
:::
::::::

::: notes
Two defects per thousand parts, and two inspectors agree maybe 80% of the time.
This is the cleanest industrial example of the ceiling from the noise slides: you
cannot score above your inspectors, and management will ask you to.
:::

## Climate and weather

:::::: columns
::: column
**Medium-range global forecasting**

- **In:** the atmosphere on a grid — pressure, temperature, wind, now and 6 h ago
- **Out:** the same grid, up to 10 days ahead
- **Fails on:** shift — the training distribution is itself moving
:::
::: column
![](img/task_climate.png)
:::
::::::

::: notes
DeepMind's GraphCast (2023) beat the operational physics model on most verification
scores, which is the result that got attention. The part that matters here is the
caveat: it is trained on a reanalysis of the past, and the climate is not stationary,
so what it learned is drifting under it by construction. Nobody has a clean answer.
:::

## Energy

:::::: columns
::: column
**Grid load and renewable output**

- **In:** the last 48 half-hours of demand, plus the weather forecast
- **Out:** demand for every half-hour of tomorrow
- **Fails on:** missingness — sensors drop out during storms
:::
::: column
![](img/task_energy.png)
:::
::::::

::: notes
Forecast wrong by 2% and someone starts a gas turbine.
The missingness point is the good one: the outages are not random, they cluster in
precisely the conditions you most need to forecast. MNAR, on a power grid.
:::

## Transport

:::::: columns
::: column
**Arrival times across a road network**

- **In:** the network, plus live speed on every segment
- **Out:** arrival time, with an interval
- **Fails on:** the long tail — the rare case is the only one that matters
:::
::: column
![](img/task_transport.png)
:::
::::::

::: notes
Ninety-nine percent of driving, and of routing, is trivially easy, and the entire
problem is the remaining fraction. This is why aggregate accuracy is meaningless
here - the same argument as the imbalance slides, told about safety instead of
screening.
:::

## Sport

:::::: columns
::: column
**Expected goals from tracking data**

- **In:** one shot — position, defenders, body part, phase of play
- **Out:** P(goal) for that shot
- **Fails on:** small n, and you only see the shots that were taken
:::
::: column
![](img/task_sport.png)
:::
::::::

::: notes
A whole season is a few hundred shots per team - genuinely small data, which is why
the arguments about xG are statistical arguments. The selection point is the deeper
one: you never observe the shot the player decided not to take.
:::

## Public sector

:::::: columns
::: column
**Risk scoring for benefits and tax**

- **In:** one household's claim — income, composition, history
- **Out:** a risk score, and a queue for investigators
:::
::: column
![](img/task_public.png)
:::
::::::

::: notes
Take this one slowly. The model was trained on who had been investigated and found
at fault before, so it learned who gets investigated - and nationality was among the
inputs. Its output then generated the next round of investigations, which became the
next round of training data.
This is the one domain on the list where the failure ended careers and ruined lives,
and it is worth saying that the technical error is one they can now name.
:::

## Which failure mode?

Try to identify what can fail:

1. A demand model was accurate for two years, then a competitor opened. Nobody retrained.
2. A defect classifier scores 99.8%. The line makes two bad parts per thousand.
3. A protein model is brilliant on a random split, mediocre on newly solved structures.
4. A hiring model trained on ten years of decisions reproduces those decisions exactly.

::: notes
Take all four before revealing. These are deliberately the same four mistakes from
this morning, wearing different clothes - if the room names them quickly, section 1
worked.
:::

## Which failure mode?

1. Competitor opens — **distribution shift**
2. Defect classifier — **class imbalance**
3. Protein model — **leakage**
4. Hiring model — **past decisions**, not merit

::: notes
Number four is the one to dwell on: there is no bug in it. The data is accurate, the
split is clean, the metric is fine, and the model is a faithful reproduction of who
got hired before. Ask what you would even measure to catch it. That question is
Week 14.
:::

# 4. What all of these have in common

## The cost of an error is usually not symmetric

| Field | A false positive costs | A false negative costs |
|---|---|---|
| **Card fraud** | a blocked card in a queue | the value of the fraud |
| **Manufacturing** | a good part scrapped | a recall |
| **Benefits fraud** | a family wrongly accused | some money |
| **Cancer screening** | a needless biopsy | a death |

::: notes
Row three is the Dutch case, and it is the row where the institution had the ratio
backwards. Ask the room to fill in a fifth row for their own final project before
they write a line of code - if they cannot, they do not yet have a problem.
This is Week 4: net benefit and decision curves are how you put this in a number.
:::

## Feedback loops

![](img/feedback.png)

::: notes
Google Flu Trends from the autopsy slide, the hiring model two slides ago, the Dutch
benefits model, and every recommender ever shipped. Once a model acts, the data stops
being a sample of the world and becomes a sample of the world your model made.
None of the theory in this course covers that arrow. Week 13 on causality is the
closest we get.
:::

## Where this course goes

- **Weeks 2–4:** the foundations, on tabular data — every field on that list
- **Weeks 5–6:** time series — energy, finance, monitoring
- **Weeks 7–10:** images and volumes — manufacturing, satellites, radiology
- **Weeks 11–12:** text — support tickets, filings, clinical notes
- **Weeks 13–14:** causality, bias, privacy, deployment — the public-sector slide

::: notes
Read the right-hand halves out loud. Every block of the syllabus was on the tour, so
nothing here is healthcare-specific machinery.
:::

# 5. Wrapping up

## Sanity checklist for any new dataset

1. What is one row?
2. Where does the label come from, when is it known, and who is it wrong about?
3. Which values are impossible, and which are in the wrong unit?
4. What is missing, and why?
5. What is the base rate?
6. How do I split so leakage is impossible?
7. What would a trivial baseline score?

::: notes
Tell them to photograph this slide. It is the last section of the recitation notebook and
they will answer all seven against MIMIC before they leave.
:::

## Apply it

> A vendor offers you 40 000 chest X-rays from three hospitals, labelled
> pneumonia / no pneumonia by a model that read the radiology reports.

Work through the seven. Which ones can you even answer?

::: notes
Three minutes in pairs, then collect answers on questions 1, 2 and 6 only - that is
enough to make the point and it fits the time.
The answers you are steering towards: one row is ambiguous (a study? a patient? a
visit?); the label is a model reading a human's summary of the image, so it is two
proxies deep; and without the hospital you cannot split by site, which after this
morning they should recognise as fatal. Somebody usually asks whether you can recover
the site from the images - yes, and that is exactly the problem.
:::

## Recitation today

**MIMIC-IV structure and data exploration**

- Download the open-access MIMIC-IV Clinical Database Demo (100 patients)
- Map the `hosp` and `icu` schemas: what joins to what, on which key
- Quantify missingness, imbalance, and irregular sampling
- Find at least one leakage trap and one ICD-version trap
- Notebook: `Course01/notebook.ipynb`

## For next week

- **TO DO:** finish the recitation notebook, including the open questions at the end
- **Next lecture:** advanced feature engineering, and missing-ness done properly


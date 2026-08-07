---
title: "Session 1 — Course Introduction & Healthcare Data"
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

![](img/noise.png)

::: notes
Left: the labels you wish you had. Right: the same 600 points, but the annotator is
cautious and calls 35% of the negatives positive. The grey line is the truth, the
amber line is what least squares gives you. Ask the room: which way did it move,
and why is that worse than random flipping?
:::

## Errors in features

- Sensor artefacts: a heart rate of 300 bpm is a detached lead, not tachycardia
- Unit chaos: weight in kg *and* lb in the same column; glucose in mg/dL *and* mmol/L
- Copy-forward: the same "current" value duplicated across days of notes
- Timestamps: charted at the time of *entry*, not the time of *measurement*

## Errors in features: three lines that find them

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

![](img/missingness.png)

::: notes
Left, patients sorted by how unwell they are: routine observations are almost always
there, the specialised ones only for the sick. Right: the patients who had a lactate
drawn died five times more often — and that is before anyone looked at the number.
Impute it away and you delete the strongest signal in the table.
:::

## Class imbalance

- Rare disease screening: 1 in 10 000

    A model that predicts "no" always is 99.99% accurate

- Blood types are not equally spread on the population

    The imbalance is changing according to the world regions

    => Can lead to distribution shift

## Class imbalance

![](img/imbalance.png)

::: notes
This model has no parameters — it is `return 0`. It beats most first attempts on
accuracy and it would kill everyone it was meant to save. Every metric we use from
Week 3 onwards exists to make this model score zero.
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

![](img/shortcut.png)

::: notes
Every "pneumonia" scan in this hospital was shot at the bedside, so the portable
unit's marker is in the corner of every positive. The model finds it in one epoch,
scores perfectly, and collapses the moment you deploy it anywhere else.
The marker here is drawn on — the phenomenon is Zech et al. 2018.
:::

## Leakage

Information at training time that will not exist at prediction time.

Typically:

- The same patient in both train and test (e.g. duplicated)
- Feature that is a deterministic function of the outcome
  (e.g. `discharge_location == 'DIED'`)
- Bug in the train/test split

## Leakage: the split that lies to you

```python
# WRONG - scaler already saw the test set
X = StandardScaler().fit_transform(X)
X_tr, X_te = train_test_split(X, test_size=.2)

# WRONG - one patient, rows in both halves
train_test_split(df, test_size=.2)

# RIGHT - split by patient, fit on train
tr, te = next(GroupShuffleSplit(test_size=.2)
    .split(df, groups=df.subject_id))
scaler = StandardScaler().fit(X[tr])
```

::: notes
The first bug leaks the test distribution through the mean and variance. The second
leaks the patient. Both look like excellent results, which is exactly why they survive
code review. You will write the third version in today's recitation.
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

# 4. Wrapping up

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

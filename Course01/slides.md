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

## Errors in features

- Sensor artefacts: a heart rate of 300 bpm is a detached lead, not tachycardia
- Unit chaos: weight in kg *and* lb in the same column; glucose in mg/dL *and* mmol/L
- Copy-forward: the same "current" value duplicated across days of notes
- Timestamps: charted at the time of *entry*, not the time of *measurement*

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

## Class imbalance

- Rare disease screening: 1 in 10 000

    A model that predicts "no" always is 99.99% accurate

- Blood types are not equally spread on the population

    The imbalance is changing according to the world regions

    => Can lead to distribution shift

## Distribution shift

Your model is trained on one slice of the world, then deployed in another:

- **Covariate shift:** P(X) changes: new hospital, new scanner, new population
- **Label shift:** P(Y) changes: a new variant, a new season
- **Concept drift:** P(Y | X) changes: treatment guidelines are updated

## Shortcuts

Models learn whatever correlates with y in *the* data.

- Pneumonia detectors that learned the **portable X-ray marker** (sicker patients get
  bedside imaging)
- Skin lesion classifiers that learned the **surgical ruler** placed next to malignant lesions
- Portrait to age models looking at **ears**

::: notes
Zech et al. 2018 (cross-site CNN generalisation); Esteva/ISIC ruler artefact.
These come back in Week 7.
:::

## Leakage

Information at training time that will not exist at prediction time.

Typically:

- The same patient in both train and test (e.g. duplicated)
- Feature that is a deterministic function of the outcome
  (e.g. `discharge_location == 'DIED'`)
- Bug in the train/test split

## Sanity checklist for any new dataset

- What is one row?
- Where does the label come from, and *when* is it known?
- What is missing, and why?
- What is the base rate?
- How do I split so that leakage is impossible?
- What would a trivial baseline score?

*You will apply this checklist in today's recitation.*

# 2. Data types and example

## The landscape

| Type | Example source | Modality |
|---|---|---|
| **EHR** | MIMIC-IV, eICU | tabular + time series + text |
| **Claims** | CMS Medicare, SEER | tabular, coded, longitudinal |
| **Imaging** | NIH Chest X-ray, BraTS | 2D / 3D arrays |
| **Waveforms** | PhysioNet ECG, EEG | high-frequency signals |
| **Genomics** | TCGA, UK Biobank | very wide, very few rows |
| **Registries / surveys** | NHANES, CDC | curated, population-level |

## EHR: electronic health records

Demographics, admissions, transfers

- Labs, vitals (irregularly sampled time series)
- Medications, orders, procedures
- Free-text notes
- ICU monitoring at minute resolution

**Strength:** rich and longitudinal.

**Weakness:** biased by care processes.

## Claims data

Generated for **billing**.

- Complete coverage across providers (you see the whole patient journey)
- Coded: ICD, CPT, HCPCS, NDC
- No clinical values — no labs, no vitals, no notes
- **Upcoding**: codes are chosen to maximise reimbursement, not to describe reality
- Lag of months

Good for: cohort size, health economics, causal studies. Bad for: physiology.

## Medical imaging

- **2D:** X-ray, fundus photography, dermoscopy, histopathology slides (gigapixel!)
- **3D:** CT, MRI — volumes, not pictures
- Stored as **DICOM** (pixels + a huge metadata header)
- Labels are scarce and expensive; segmentation masks even more so
- Weeks 7–10

## Waveforms and signals

- ECG (250–500 Hz), EEG, arterial blood pressure, PPG
- Enormous volume, tiny labelled fraction
- Artefact-dominated: movement, disconnection, electrical noise
- Weeks 5–6

## Genomics & omics

- p ≫ n: 20 000 genes, 300 patients
- Batch effects dominate biological signal if you are careless
- Multiple testing everywhere
- Week 12 (survival on TCGA)

## Multi-modality is the real world

A single patient generates *all* of these simultaneously.

- MIMIC-IV (tabular + notes) ↔ MIMIC-CXR (images) — linked by `subject_id`
- Fusion strategies: early / late / joint embedding
- Week 8: CLIP-style image–text pairing in radiology

# 4. Wrapping up

## Key takeaways

- Real data violates every assumption of the textbook setup
- **Noise, missingness, imbalance, shift** — name them explicitly for every project
- In healthcare you observe **care processes**, not biology
- Missingness carries information — and leakage
- Standards (FHIR, OMOP) and vocabularies (ICD, LOINC, SNOMED) are what make data joinable
- Always ask: what does one row mean, and when is the label known?

## Recitation today (1.5h)

**MIMIC-IV structure and data exploration**

- Download the open-access MIMIC-IV Clinical Database Demo (100 patients)
- Map the `hosp` and `icu` schemas: what joins to what, on which key
- Quantify missingness, imbalance, and irregular sampling
- Find at least one leakage trap and one ICD-version trap
- Notebook: `Course01/notebook.ipynb`

## For next week

- **Read:** Johnson et al., *MIMIC-IV, a freely accessible electronic health record dataset*
  (Scientific Data, 2023)
- **Skim:** Rubin (1976) on missing data mechanisms — we build on it directly
- **Do:** finish the recitation notebook, including the open questions at the end
- **Next lecture:** advanced feature engineering, and missingness done properly

## Questions?

Office hours: by appointment

`Course01/` on the course repository

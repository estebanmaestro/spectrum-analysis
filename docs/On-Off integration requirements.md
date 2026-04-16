# **Statistical Rule-based vs Clustering ON/OFF detection — Implementation guideline**

**Audience:** EMI Modeling: Running-state detection when **existing ON/OFF labels** vary in **quantity**, **class balance**, and **operational diversity**.  
**Context:** The Running state model integrates a weighted 5-feature scoring approach (the selected rule-based method) with clustering using high-dimensional spectrum features.

---

## **1\. Introduction and Assumptions**

The EMI running state model considers two components:

1. ***Clustering method*** with an On/Off state determination.  
   1\. Site-level separability (0 – 100). Measures how cleanly the two clusters are separated on the log-power axis. Higher is better; a target of ≥ 70 is recommended.  
   2\. Per-prediction score (0 – 100). For every individual spectrum, the distance of its log-power value to the midpoint boundary between the two cluster means is normalised by the half-gap. Higher is better; a floor of ≥ 35 is recommended.  
2. ***Statistical Rule-based approach*** that compares 5 key features having strong statistical association with the On/Off states and computes a weighted voting using these features to generate a ***Running State Index***. This index can be compared with a ***global*** or ***asset-dependent threshold*** to infer the running state.

**NOTE:** This document covers the simple case in which assets are modeled independently and direct/indirect associations between multiple asset running states are not considered.

---

## **2\. Methodology**

### **2.1 Dataset Size (total available spectra)**

Use **per-asset** counts. If multiple units are available for the same site, each asset should have independent evaluation based on its registered spectra.

| Tier | Total *n* (labeled) | Approach |
| :---: | :---: | :---: |
| **Small** | *n* \< 10 | Statistical validation is weak; use the general index with a predefined global threshold. |
| **Medium** | 10 ≤ *n* \< 100 | Acceptable for statistical approach, rule-based is preferred, an asset dependent threshold can be adjusted and results can be validated using clustering. |
| **Large** | *n* ≥ 100 | Robust estimates for both procedures, clustering approach is recommended. |

Also track ***per-class*** counts *n*ON, *n*OFF; the limiting factor is usually **min(*n*ON, *n*OFF)** for binary problems.

### **2.2 Class imbalance**

Let ***r \= max(nON, nOFF) / min(nON, nOFF)***

(define *r* \= ∞ if one class is absent).

| Band | *r* | Notes |
| :---: | :---: | :---: |
| **Balanced** | *r* ≤ 1.5 | Robust estimates for both procedures, clustering approach is recommended. |
| **Moderately imbalanced** | 1.5 \< *r* ≤ 4 | Acceptable for statistical approach, rule-based is preferred, an asset dependent threshold can be adjusted and results can be validated using clustering. |
| **Strongly imbalanced** | 4 \< *r* ≤ 15 | Statistical validation is weak; use the general index with a predefined global threshold |
| **Extremely imbalanced** | *r* \> 15 or minority \< 5 | Statistical validation is weak; use the general index with a predefined global threshold |

---

## **3\. Available methods — what they are and how they relate**

This project uses two complementary methods for ON/OFF detection. It is important to understand what each one actually does before deciding when to deploy them.

### **3.1 Weighted 5-Feature Scorer (`OnOffScorer`) — the selected rule-based method**

**What it is:** A composite weighted-voting system implemented in `src/on_off_scorer.py`. For each spectrum, five spectral features are computed and each one casts a binary vote (ON or OFF) by comparing its value to a **median threshold**. The votes are multiplied by learned weights and summed into a **Running State Index** (normalised to [0, 1]). That index is compared to a **decision threshold** (default 0.35) to produce the final ON/OFF prediction.

**The five features and their weights (default configuration):**

| Feature | Weight | Direction (vote ON when …) |
| :--- | :---: | :--- |
| `ssn_B1` — normalised spectral slope in band B1 (30–300 kHz) | 0.945 | value ≥ median |
| `lr_B2_B4` — log-ratio of B2 to B4 band power | 0.839 | value ≥ median |
| `spectral_spread` — global spectral spread | 0.811 | value < median |
| `rolloff_norm_B3` — normalised 85%-energy rolloff in B3 | 0.738 | value < median |
| `curvature_B2` — 2nd-order log-log curvature in B2 | 0.719 | value ≥ median |

**Two operating modes** arise from the median thresholds:

* **Global medians** (the defaults, learned from a 272-spectrum training set): used when the asset has little or no site-specific labeled data.
* **Asset-specific medians**: computed from the labeled spectra of a particular asset/site once enough data is available, passed via the `medians` constructor parameter.

The scorer also computes `spectral_flatness_global` and flags spectra above a flatness threshold (default 0.40) as **flat-line** — a quality indicator that warns when shape-based features are unreliable.

| Strength | Limitation |
| :--- | :--- |
| Interpretable: every vote and weight is auditable | Median thresholds from global defaults may not fit assets with unusual spectral profiles |
| Works immediately on new assets with zero site data (global mode) | Binary per-feature votes lose granularity (a value barely above median counts the same as far above) |
| Lightweight, deterministic, no fitting step at inference time | Cannot capture nonlinear interactions between features |
| Degrades gracefully: with small data, only the decision threshold or medians need tuning, not the full model | Weights are fixed; if the relative importance of features shifts for an asset class, the weights need re-derivation |

### **3.2 Clustering-based ON/OFF (unsupervised binary partition)**

**What it is:** Fit a **two-group** structure in feature space (k-means k=2, Gaussian mixture with two components, spectral methods, etc.), then **assign** which cluster is ON using labels (even a handful) or SME direction.

Each cluster produces two quality metrics:

1. **Site-level separability** (0–100): how cleanly the two clusters separate on the log-power axis. Higher is better; ≥ 70 is a good target.
2. **Per-prediction confidence** (0–100): distance of an individual spectrum to the cluster boundary, normalised by the half-gap. Higher is better; ≥ 35 is a reasonable floor.

| Strength | Limitation |
| :--- | :--- |
| Uses unlabeled structure; discovers data-driven boundaries | With **imbalance**, the larger class can **absorb** minority structure |
| Can track slow drift if refit on a schedule | **Label–cluster inversion** is possible; always resolve with labels or SME |
| Captures multivariate separation that no single feature threshold can | **High diversity + small *n*** produces unstable partitions |
| Provides a confidence metric per prediction | Requires enough data for stable cluster formation |

**Data needs:** Meaningful when **min(*n*ON, *n*OFF)** is not trivial and clusters are **coherent** under bootstrap. With **micro *n***, treat clustering as **exploratory**; report stability and direction checks, not production accuracy.

### **3.3 How the two methods complement each other**

The weighted scorer and the clustering approach are **not** competitors — they answer different questions and have different data requirements. The decision of when to use each (or both) depends on the labeled data at disposal.

| Data situation | Primary method | Role of the other method |
| :--- | :--- | :--- |
| **Very small data** (*n* < 10) or no site labels | **Weighted scorer with global medians** | Clustering is exploratory only — do not use its output for decisions |
| **Small data** (10 ≤ *n* < 40), adequate balance | **Weighted scorer with asset-specific medians** (recompute medians from site data) | Clustering as **validation**: compare scorer predictions vs cluster assignments to detect disagreement |
| **Medium data** (40 ≤ *n* < 150), adequate balance | **Both**: weighted scorer as interpretable baseline, clustering as primary if separability ≥ 70 | When both agree, confidence is high. Log disagreements for SME review |
| **Large data** (*n* ≥ 150), adequate balance | **Clustering is recommended** as primary with scorer as guardrail | Weighted scorer serves as a **sanity check** and fallback if cluster quality degrades |
| **Any size, strong imbalance** (*r* > 4) | **Weighted scorer** (more robust to imbalance) | Clustering is unreliable under severe imbalance — use only for exploratory analysis |
| **External SCADA truth available** | SCADA-derived labels are **primary ground truth** | Both methods are validated against SCADA; use the one with best agreement |

**The typical progression for a new site:**

1. **Day zero** — no site-specific labels: deploy the weighted scorer with **global medians** and default threshold (0.35). This produces an immediate Running State Index.
2. **Early data** (< 40 labeled spectra): re-derive **asset-specific medians** from site data; optionally adjust the decision threshold. Run clustering **in shadow** to check if clusters emerge.
3. **Sufficient data** (40+ labeled spectra, reasonable balance): activate clustering alongside the scorer. Use **disagreement logging** — when scorer and clustering disagree on a spectrum, flag it for review. If site-level separability ≥ 70, clustering can become the primary method.
4. **Mature data** (150+ spectra with good balance): clustering is the recommended primary method. Keep the weighted scorer running as an **interpretable guardrail** and drift detector.

### **3.4 Leveraging site-specific labeled data**

When a site accumulates labeled spectra with reasonably clear ON/OFF assignments, that data can be used to improve both methods. The question is *how much* the data should inform each method, and whether clustering alone is sufficient or the scorer should also be recalibrated.

#### **What site-specific data gives you**

The labeled spectra at a site provide two things that the global defaults cannot:

1. **Site-specific feature distributions.** The ON and OFF populations at a particular asset may have different median values for the five features than the global training set. For example, a site with high ambient noise may shift `ssn_B1` upward for both classes, making the global median a poor separator.
2. **Site-specific power-level separation.** The raw per-band power levels (B1–B4) and the total log-power often show a clean gap between ON and OFF at a given site — sometimes cleaner than the derived features. If the labels are trustworthy, this gap is directly observable in the data and can be quantified.

#### **Recalibrating the scorer with site data**

The `OnOffScorer` is designed for this. When enough labeled spectra are available:

1. **Recompute medians per feature.** For each of the five features, take the **midpoint between the ON-class median and the OFF-class median** at that site. Pass these as the `medians` parameter. This shifts the vote boundaries to reflect the actual class-conditional distributions at the asset.
2. **Adjust the decision threshold.** With site-specific medians, the optimal threshold may differ from the global 0.35. Use leave-one-out or a small holdout to find the threshold that best separates the labeled data (targeting the desired precision-recall trade-off).
3. **Validate feature relevance.** Not all five features may be equally informative at every site. Compute the **per-feature effect size** (e.g., Cohen's *d* between ON and OFF distributions) at the site. If a feature shows no separation (*d* < 0.2), it contributes noise to the index; consider down-weighting or excluding it via the `feature_params` parameter.

**Minimum data for recalibration:** asset-specific medians become meaningful with **≥ 10 labeled spectra per class** (so *n* ≥ 20 with reasonable balance). Below that, the class medians are unstable and global defaults are safer.

#### **Using per-band power levels directly**

The per-band power levels (B1, B2, B3, B4 total power and the global log-power) are the raw inputs from which both methods derive their signals. When labels are clear, there is value in examining these directly:

* **Diagnostic check:** plot the log-power distribution by label. If there is a **visible gap** between ON and OFF distributions with minimal overlap, this confirms the site has good separability — both the scorer and clustering should work well.
* **Simple power-level threshold as a fast sanity check:** if ON spectra consistently have higher total power than OFF spectra (or vice versa) with a clear gap, a single power-level threshold can serve as a **quick cross-validation** for both the scorer and clustering. This is not a replacement for either method but a useful diagnostic.
* **Feeding power levels into clustering:** the clustering approach already operates on log-power or derived features. If the raw band powers show clean separation, clustering will naturally capture it. The advantage of clustering over a raw power threshold is that it handles multivariate structure — cases where no single band cleanly separates the classes but the combination does.

#### **Is clustering alone sufficient when labels are clear?**

**Not always.** Clustering is strongest when it discovers structure that aligns with the labels, but it has failure modes even with clear data:

| Situation | Clustering alone | Recommended approach |
| :--- | :--- | :--- |
| Clear power-level gap, balanced data, *n* ≥ 40 | **Yes** — clustering will find the gap and produce high separability scores | Clustering as primary, scorer as guardrail. Both will agree. |
| Clear labels but **imbalanced** data (*r* > 4) | **Risky** — k=2 may place the boundary too close to the minority class, absorbing edge cases | Scorer with **site-specific medians** as primary; clustering as secondary validation only. |
| Clear labels but **multimodal** ON or OFF (e.g., different operating regimes) | **Risky** — k=2 may split along the wrong axis (regime boundary instead of ON/OFF boundary) | Scorer as primary (it aggregates across features and is not confused by multimodality). Clustering with k > 2 + label-guided merge as exploratory. |
| Clear labels but **small *n*** (10–30) | **Unstable** — cluster centroids and boundaries shift significantly with small perturbations | Scorer with **site-specific medians** as primary. Clustering in shadow mode to monitor stability. |
| Clear labels, large balanced data, but **low separability** on power alone (features overlap) | **Weak** — clustering produces low separability scores and low per-prediction confidence | Neither method will be highly accurate. Scorer is more robust here because it combines five features with learned weights. Report the Bayes error expectation honestly. |

**The practical answer:** site-specific labeled data should **always** be used to recalibrate the scorer (at minimum, recompute medians). Clustering should be activated alongside it when the data volume and balance justify it. The two methods together are stronger than either alone — the scorer provides an interpretable, stable baseline while clustering captures data-driven structure that the scorer's fixed features might miss.

---

## **4\. Scenario playbook (existing ON/OFF labels)**

Each row assumes **binary** ON/OFF unless noted. **Scorer** = weighted 5-feature approach (`OnOffScorer`); **Cluster** = two-cluster model in feature space. The guidance specifies whether the scorer should use **global** or **asset-specific** medians and when clustering adds value.

### **4.1 Micro quantity (*n* < 12)**

| Scenario | Balance | Diversity | Weighted scorer guidance | Clustering guidance | Evidence stance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **M-B-L** | Balanced | Low | Use **global medians**. If the few available labels show a clear gap in the top-weighted features (ssn_B1, lr_B2_B4), consider a **conservative** threshold adjustment; document sensitivity to single outliers. | k=2 **exploratory** only; **bootstrap stability** mandatory; high risk of arbitrary splits. Do not override scorer decisions. | **Pilot / not for automated enforcement.** |
| **M-B-H** | Balanced | High | Use **global medians**; do **not** attempt asset-specific tuning — data cannot cover the operational envelope. If external SCADA ON/OFF is available, prefer it as primary. | **Do not** claim a site-wide ON cluster; subpopulations are under-sampled. | **Research only** until more labeled regimes exist. |
| **M-I-L** | Imbalanced | Low | Use **global medians**. The scorer tolerates imbalance better than clustering. If the minority class is spectrally separated on the dominant feature, a **minority detector** threshold on ssn_B1 alone may be trialed. | k=2 often **collapses** minority into majority cluster; if used, analyze **cluster purity** only descriptively. | Report **recall on minority** with wide confidence intervals. |
| **M-I-H** | Imbalanced | High | Use **global medians** with extreme caution; imbalance + diversity with micro data makes any method fragile. | Same as M-I-L, amplified. | **Block** production automation without more data or external truth. |

### **4.2 Small quantity (12 ≤ *n* < 40)**

| Scenario | Balance | Diversity | Weighted scorer guidance | Clustering guidance | Evidence stance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **S-B-L** | Balanced | Low | Good case for **asset-specific medians** derived from site data. Use **leave-one-out** on the Running State Index to test threshold stability. All 5 features contribute. | k=2 with **reduced features** as **validation**: check **agreement rate** between scorer and cluster assignments; report **cluster flip** under perturbations. | **Limited production** possible with SME sign-off and monitoring. |
| **S-B-H** | Balanced | High | Compute **asset-specific medians** but keep thresholds **conservative** (wider than the strict class boundary). Avoid per-season tuning unless each stratum has ≥ 10 rows. | Consider **stratified** clustering within operating regimes if *n* per stratum is viable; else treat as medium-diversity in documentation. | **Shadow mode** recommended before automation. |
| **S-I-L** | Imbalanced | Low | Scorer is preferred. Compute **asset-specific medians** but apply **cost-sensitive** threshold tuning — lower the threshold to improve minority recall. Validate minority recall explicitly. | Try **supervised** assignment of clusters (map cluster to ON/OFF using labeled rows) but expect **variance**; prefer **anomaly vs normal** framing if minority is OFF. | **Conditional go** if minority recall meets a prewritten bar on LOO-CV. |
| **S-I-H** | Imbalanced | High | Use **global medians** as anchor; asset-specific tuning is risky with imbalanced high-diversity small data. Prefer **external** labels for threshold calibration if available. | Avoid single global k=2; pooling may hide minority regimes. | **Research / shadow** until balance improves per stratum. |

### **4.3 Medium quantity (40 ≤ *n* < 150)**

| Scenario | Balance | Diversity | Weighted scorer guidance | Clustering guidance | Evidence stance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Med-B-L** | Balanced | Low | **Asset-specific medians** are well-supported. Scorer can be **benchmarked** against clustering; if both perform within noise, prefer the scorer for interpretability. | Full or **subset** features; use **train/holdout** (time-aware if sequential). If separability ≥ 70, clustering becomes a viable **primary** method. | **Production candidate** with monitoring. |
| **Med-B-H** | Balanced | High | Use **asset-specific medians** with **robust** global thresholds (quantiles). Avoid per-segment tuning unless *n* per segment supports it. | Consider **mixture** with k > 2 **only** for exploratory regime detection; binary decision still needs explicit collapse policy. | **Production candidate** with drift monitoring per segment. |
| **Med-I-L** | Imbalanced | Low | Scorer remains robust. Combine **asset-specific medians** with **recall-oriented** threshold tuning on the minority class. | Use **balanced sampling** or **weighted** k-means/GMM; validate minority class metrics. Clustering is an **assist** here, not primary. | **Production candidate** with explicit minority SLO. |
| **Med-I-H** | Imbalanced | High | Scorer per **stratum** (if *n* supports) or use **regularised** threshold adjustment. | Clustering as **assist**: identify subclusters, label with SME, then revisit. | **Staged rollout**; revisit after more minority regimes are captured. |

### **4.4 Large and very large quantity (*n* ≥ 150)**

| Scenario | Balance | Diversity | Weighted scorer guidance | Clustering guidance | Evidence stance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **L-B-L / VL-B-L** | Balanced | Low | Scorer serves as **interpretable baseline** and **guardrail** around the cluster model. Keep running it in parallel. | Strong k=2 baselines; ensembles can reduce variance. **Clustering is the recommended primary method.** | **Production standard** with periodic refit policy. |
| **L-B-H / VL-B-H** | Balanced | High | **Monitor** scorer drift per regime; consider scorer as a **hierarchical** guard (coarse ON/OFF then cluster refines sub-state). | k=2 per **regime** or **hierarchical clustering** + labeled assignment per leaf if *n* per leaf is sufficient. | **Production standard**; invest in **continuous validation** buckets. |
| **L-I-L / VL-I-L** | Imbalanced | Low | Scorer with **PR-optimised** threshold remains a strong option even at large *n*; calibrate thresholds on precision-recall, not accuracy alone. | Use **class-informed** fitting or **one-vs-rest** anomaly detection for minority if minority is the critical miss. | **Production standard** with minority-focused alerts. |
| **L-I-H / VL-I-H** | Imbalanced | High | Often needs a **model portfolio**: scorer for the obvious majority, **specialised** detection for rare ON or OFF regimes. | Same: **multi-cluster** structure with **supervised** mapping from clusters/regions to ON/OFF. | **Production** with **rare-event** review workflow. |

---

## **5\. Additional cases (labels or physics are non-ideal)**

| Case | Description | Weighted scorer | Clustering | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **No labels** | Only spectra, no ON/OFF truth | Use **global medians** — the scorer still produces an index; treat output as **provisional** | Exploratory partitions; align with SME or SCADA later | Output is **structure**, not validated ON/OFF. |
| **One-class labels** | Only ON or only OFF known | Use the known class to validate scorer medians for that class; detect **departure** from the known class envelope | One-class / density models; weak binary story | Frame as **novelty detection**, not balanced accuracy. |
| **Noisy labels** | Confirmed errors or weak SCADA | Use **robust** medians (trimmed means); avoid fitting the threshold to single outliers | **Noise-tolerant** losses if supervised; else clean a **core** set first | Document **assumed label error rate**. |
| **Multimodal ON (or OFF)** | One label, multiple spectral clusters | Global scorer threshold often works (the index aggregates across features) but may lose sharpness | k > 2 then **merge** clusters by label majority vote per cluster | Requires enough *n* to **stabilise** subclusters. |
| **Heavy class overlap** | Labels correct but features non-separable | Scorer will produce scores near the threshold for most spectra; **accept Bayes error** and report the uncertainty | Unsupervised binary will **hurt**; prefer **probabilistic** supervised models or more informative features | Set expectations with stakeholders **before** modeling. |
| **Pooled multi-unit labels** | One model across units | Scorer with **global medians** is a natural pooled method; only refine asset-specific medians per unit if *n* supports it | **Per-unit** clustering usually safer unless *n* per unit is micro | If pooled, report **per-unit** confusion where possible. |
| **Temporal drift** | Process or sensor changes | **Schedule** median and threshold reviews on a cadence; compare scorer output to a frozen baseline in shadow | **Scheduled refit**; compare to frozen baseline in shadow | Tie to **change-point** monitoring on key features. |

---

## **6\. Evaluation discipline (ties quantity, balance, and diversity together)**

| Quantity tier | Primary evaluation tactic | Metrics to emphasize |
| :--- | :--- | :--- |
| **Micro** | Leave-one-out on scorer index; **bootstrap** clustering stability (exploratory) | Descriptive separation, **wide** CIs; avoid single F1 point estimates |
| **Small** | Repeated stratified splits or LOO for scorer threshold; clustering agreement rate | Balanced: accuracy/F1; Imbalanced: **minority recall**, PR-AUC |
| **Medium+** | Holdout + (optional) k-fold with **time** respect if sequential; scorer vs cluster disagreement rate | F1, MCC, **calibration** curves if probabilities exist |
| **Large+** | Holdout + **temporal** backtest buckets; scorer as interpretable audit trail | Same + **slice** metrics per diversity stratum |

**Imbalance:** always report **confusion matrix** and **per-class** precision/recall; headline accuracy is misleading.

**Diversity:** report metrics **global** and **per stratum** (when *n* per stratum allows); if strata are too thin, disclose **coverage gaps** instead of silent averaging.

**Scorer–Cluster agreement:** when both methods are active, report the **agreement rate** and characterise **disagreement cases** (are they near the threshold? in a particular operating regime?). Persistent disagreement patterns indicate either scorer medians need recalibration or cluster stability is poor.

---

## **7\. Minimum documentation per site (short checklist)**

* Quantity tier and (*n*ON, *n*OFF), balance ratio *r*, diversity level.
* Feature set version and whether **global** or **asset-specific** medians are used.
* Scorer threshold in use (default 0.35 or site-tuned value), and rationale for any adjustment.
* Chosen path: **scorer only** (global or asset-specific), **scorer + clustering validation**, or **clustering primary + scorer guardrail**, with rationale from §3–§5.
* Whether clustering is active, and if so: site-level separability and per-prediction confidence ranges.
* Evaluation protocol and **known weaknesses** (for example "minority OFF only observed at one load").
* Production stance: **research**, **shadow**, **conditional**, or **standard**, with monitoring hooks.
* **Disagreement log** policy: how scorer-vs-cluster disagreements are reviewed and resolved.

---

## **Related repository artifacts**

* `src/on_off_scorer.py` — production implementation of the weighted 5-feature scorer (`OnOffScorer` class).
* `HANDOFF.md` — feature pipeline and per-site clustering context.
* `_run_unsupervised_v3.py` — multi-method clustering and reporting.
* `_validate_aep_turk.py` — pattern for **limited or unknown** labels and directional checks.

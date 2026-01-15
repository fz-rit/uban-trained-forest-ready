# 🧭 **Urban-Trained, Forest-Ready Domain Adaptation Pipeline - v1**

#### **Step 1 – Mixed Pretraining (Source Mix)**

Goal: build a strong general backbone with broad geometry knowledge.

1. Gather datasets: **Semantic3D (urban)** + **ForestSemantic/DigiForests (synthetic/natural forests)**.
2. Normalize coordinates and intensities to a unified scale and unit.
3. Merge datasets while keeping source-domain labels consistent.
4. Train **KPConv (or RandLA-Net)** from scratch or fine-tune from pretrained weights.

   * Use cross-entropy loss on labeled points.
   * Data augmentation: random rotation, jitter, dropouts, and subsampling.
5. Save checkpoint weights (`pretrained_mix.ckpt`).

---

#### **Step 2 – Self-Supervised Warm-Up (Target Forest Data, Unlabeled)**

Goal: adapt low-level filters and normalization stats to forest domain noise.

1. Load unlabeled target datasets (e.g., **Harvard Forest**, **Palau Mangrove TLS**).
2. Choose self-supervised tasks, e.g.:

   * Point masking & reconstruction (masked autoencoder-style)
   * Rotation or transformation prediction
   * Contrastive learning between augmented views
3. Freeze backbone partially (early KPConv layers), train head(s) for a few epochs.
4. Monitor feature embedding stability using visualization (e.g., t-SNE).
5. Save checkpoint (`warmup_target.ckpt`).

---

#### **Step 3 – Unsupervised Domain Adaptation (Adversarial + Geometric Consistency)**

Goal: align features between source (mixed) and target (forest).

1. Set up **adversarial feature alignment**:

   * Add a domain discriminator (D(f)) that distinguishes source vs. target features.
   * Train discriminator and feature extractor adversarially (à la DANN).
2. Apply **local geometric consistency loss**, e.g.:

   * Neighbor smoothness (points in a local patch should share similar labels).
   * Spatial contrastive consistency.
3. Combine losses:
   [
   \mathcal{L} = \mathcal{L}*{CE}^{source} + \lambda_1 \mathcal{L}*{adv} + \lambda_2 \mathcal{L}_{geo}
   ]
4. Train until domain-confusion loss plateaus.
5. Save adapted weights (`adapted_model.ckpt`).

---

#### **Step 4 – Pseudo-Label Refinement (Self-Training)**

Goal: further boost mIoU and stabilize rare-class boundaries.

1. Use `adapted_model.ckpt` to infer labels on the unlabeled target data.
2. Filter predictions: keep only points with confidence > threshold (e.g. 0.8).
3. Treat those as **pseudo-labels** for supervised fine-tuning.
4. Retrain the model briefly (1–3 epochs) with a smaller learning rate.
5. Optionally iterate once more with updated pseudo-labels.
6. Evaluate on held-out labeled subset (if available) → report mIoU gain.

---

### ⚙️ Optional Step 5 – Evaluation & Efficiency Tracking

* Compare to baseline (urban-only KPConv).
* Record GPU memory usage, training time, and mIoU improvement.
* Visualize qualitative results (e.g., stem/canopy color maps).


---

---

# 🧭 **Urban-Trained, Forest-Ready Domain Adaptation Pipeline - v2**

* **Baseline → +UDA → +PL** on **Harvard Forest/Mangrove** style data
* Side-by-side visuals (colorized mIoU, stems/terrain overlays)
* **One small “diffusion touch”** that’s feasible in 3 weeks:

  * Use a **tiny denoising-diffusion SSL** pretext on small point patches **OR** (fallback) a **“diffusion-inspired” denoise consistency** that doesn’t require a full DDPM

---

# Week 1 — Baseline you can trust (MVP-1)

**Goal:** A mixed-source baseline with solid logs + visuals.

**Data**

* Source: **Semantic3D (urban)**
* Target: **Mangrove3D / Harvard Forest** (use what you already have; even small labeled subsets help)
* Normalize coords, intensity; unify units.

**Training**

* Backbone: **RandLA-Net (Open3D-ML)** or KPConv (pick the one you already run fastest).
* Augs: rotation (z), jitter, random drop, grid-subsample.
* Loss: CE on labeled only (source + any labeled target you have).

**Deliverables**

* `baseline_mix.ckpt`
* Confusion matrix + qualitative point-color renders (stems/terrain/canopy).

**Tip (commands)**

* Train baseline (Open3D-ML CLI or your existing script).
* Save a few **standard camera views** for A/B comparisons later.

---

# Week 2 — Light UDA + Pseudo-Label loop (MVP-2)

**Goal:** Add simple, reliable UDA without deep surgery; then self-training.

### A) Simple UDA (choose ONE to keep it lean)

1. **Entropy Minimization + Consistency** (no discriminator)

* Add **sharpened entropy loss** on target predictions.
* Consistency: two augmentations of same target patch → KL consistency.

2. **Tiny Domain Discriminator (DANN-lite)**

* A small MLP on encoder features (detach gradients correctly).
* Keep λ_adv small to avoid instability.

> Pick (1) if you want minimal code change. Pick (2) if you want a classic UDA checkbox.

### B) Pseudo-Label (PL) loop v1

* Infer on unlabeled target → filter **pmax > 0.8** → fine-tune 1–2 short epochs.
* Optionally iterate once more.

**Deliverables**

* `uda.ckpt` → `pl_v1.ckpt`
* Table: baseline vs +UDA vs +PL (overall mIoU + class F1, especially stems/terrain).

---

# Week 3 — “Diffusion touch” + polishing (MVP-3)

**Two pathways—choose P0 vs P1 based on time**

### P0 (Safest): **Diffusion-inspired denoise consistency** (no full DDPM)

* Idea: approximate the *spirit* of diffusion denoising without training a DDPM.
* For each target patch:

  1. Add small Gaussian jitter to xyz (and/or intensity).
  2. Pass original & jittered through the model.
  3. Add **consistency loss** between logits (KL or MSE on softmax).
* Use **multi-σ** jitters over batches to mimic diffusion timesteps (e.g., σ∈{0.005, 0.01, 0.02} of scene scale).
* Where to plug it: **Step 2 (warm-up)** or **Step 3 (UDA)** as an extra regularizer.
* What to say in the demo: “We apply a diffusion-inspired denoise consistency to stabilize features under noise, improving target robustness.”

### P1 (Stretch): **Tiny DDPM SSL on patches (T=50)**

* Train a **small ε-predictor** on 2–8k-point patches from target forests:

  * Encoder: PointNet-like or your RandLA-Net encoder (frozen), with a small MLP head that takes (features, timestep embedding).
  * Objective: MSE(ε_pred, ε_true) with a short schedule (T=50; cosine β).
  * Use it **only as an SSL pretext** to warm up the encoder; then fine-tune for segmentation.
* Keep it tiny (few hours training on small patches). If it drags, drop back to **P0**.

### P2 (If time remains): **PL stability filter**

* When selecting pseudo-labels, **denoise once** (either via P1’s ε-predictor for a few steps or just apply a small jitter+denoise cycle) and **keep only labels that stay the same**.
* Rule: keep point i if `pmax>0.8` **and** `argmax(logits_original)==argmax(logits_denoised)`.

**Final polishing**

* Re-render the same camera views for baseline/UDA/PL/“diffusion-touch”.
* Make one A4 slide with pipeline diagram + before/after images + a tiny table.

---

## File/Module sketch (keep it organized)

```
repo/
  data/
  configs/
  models/
    seg_backbone.py         # RandLA/KPConv wrapper
    domain_discriminator.py # (if DANN-lite)
    diffusion_ssl_tiny.py   # (only if doing P1)
  losses/
    entropy_min.py
    consistency_kl.py
    diffusion_consistency.py  # for P0
  scripts/
    train_baseline.py
    train_uda.py
    gen_pseudo_labels.py
    finetune_with_pl.py
    eval_render.py
  outputs/
    baseline_mix.ckpt
    uda.ckpt
    pl_v1.ckpt
    figs/
```

---

## Minimal code hooks (pseudo, short)

**Consistency loss (P0)**

```python
# logits_a, logits_b from two noisy versions of same target patch
cons_loss = kl_div(softmax(logits_a), softmax(logits_b)).mean()
total_loss = ce_source + λ_ent*entropy_min + λ_cons*cons_loss
```

**Domain disc (DANN-lite)**

```python
feat = encoder(points)               # [B,N,C]
feat_g = grad_reverse(feat, λ_adv)
d_logits = D(feat_g.mean(dim=1))     # global pooling + MLP
L_adv = bce(d_logits, domain_labels) # 1=source, 0=target
```

**PL select**

```python
prob = softmax(logits)
conf, pseudo = prob.max(dim=-1)
mask = conf > 0.8
train_on(points[mask], pseudo[mask])
```

---

## What to measure (simple, presentable)

* mIoU on a small held-out set (or cross-val if labels are scarce)
* **Stems F1** (your viewers will care about stems/boundaries)
* Qualitative: 3–4 consistent viewpoints with label overlays

---

## Priorities (so you ship something)

* **P0 MUST:** Baseline → +Entropy/Consistency UDA → +PL → visuals
* **P1 NICE:** Tiny DDPM SSL warm-up on patches (only if smooth)
* **P2 NICE:** PL stability filter (denoise-agree rule)

---

## Risks & fallback

* If UDA destabilizes: turn off DANN, keep **entropy+consistency** only.
* If DDPM feels heavy: stick to **P0 diffusion-inspired** consistency; it’s defensible and fast.

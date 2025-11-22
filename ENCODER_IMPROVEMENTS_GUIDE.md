# Encoder Improvement Guide: Answers to Your Questions

## 1. Will increasing steps increase overfitting risk?

### Short Answer:
**For CEBRA phases (unsupervised/supervised), increasing steps is generally safe and often beneficial.** The risk of overfitting is lower than in traditional supervised learning because:

### Why CEBRA is Less Prone to Overfitting:
- **Contrastive Learning Nature**: InfoNCE learns relative similarities between pairs, not absolute labels. This encourages the model to learn generalizable structure rather than memorize specific patterns.
- **Temporal Structure**: CEBRA-Time learns smooth temporal transitions, which are inherently generalizable across patients.
- **Label Alignment**: CEBRA-TimeDelta aligns embeddings with emotion labels, but still maintains the learned temporal structure.

### Your Specific Situation:
- **Current State**: Good training performance, poor test performance on EC304
- **Likely Cause**: The encoder hasn't learned **transferable features** that generalize to EC304, not overfitting
- **Recommendation**: 
  - **Increasing steps is likely helpful** if the model hasn't converged yet (loss still decreasing)
  - Monitor the **validation loss during CEBRA phases** - if it's still decreasing, more steps will help
  - If validation loss plateaus, more steps won't help (but won't hurt much either)

### Is Increasing Steps Necessary When Increasing Latent Dimension?
- **Not strictly necessary**, but they complement each other:
  - **More latent dimensions** = more capacity to represent complex patterns
  - **More steps** = more time to learn those patterns
  - If you increase latent_dim from 16→32, you're adding capacity, so more steps (e.g., 6000/4000) can help the model fully utilize that capacity

### Practical Recommendation:
- Start with **--unsup-steps 6000, --sup-steps 4000** when using **--latent-dim 32**
- Monitor training curves - if loss is still decreasing, you can go higher
- If loss plateaus early, you can reduce steps to save compute

---

## 2. What will increasing latent dimension lead to conceptually?

### Conceptual Effects:

#### **More Representational Capacity**
- **16D**: Can capture ~16 orthogonal features/dimensions of emotion-related neural activity
- **32D**: Can capture ~32 orthogonal features, allowing for:
  - More nuanced emotion distinctions
  - Patient-specific variations within the same emotion
  - Complex interactions between multiple neural patterns

#### **Better Separation of Emotion Classes**
- Higher dimensions allow the embedding space to create more complex **decision boundaries**
- Emotions that are similar (e.g., Fear vs. Disgust) can be separated in higher-dimensional space even if they overlap in lower dimensions

#### **Patient-Specific Nuances**
- **Key for your use case**: Different patients may express the same emotion with slightly different neural patterns
- 32D space can learn **patient-invariant emotion features** while also capturing **patient-specific variations**
- This is crucial for generalization to EC304

#### **Potential Risks**
- **Overfitting**: More parameters = more risk, but mitigated by:
  - Contrastive learning (less prone to overfitting)
  - Large aggregated dataset (~7000 timesteps)
  - Regularization (weight_decay, temporal consistency)
- **Computational Cost**: Slightly more memory and compute, but usually manageable

#### **Why 32D Might Help EC304**
- If EC304's neural patterns are subtly different from the training patients, 16D might not have enough capacity to learn a **shared embedding space** that works for both
- 32D gives the model more "room" to learn patient-invariant features while accommodating patient-specific variations

### Practical Recommendation:
- **Try --latent-dim 32** with increased steps (6000/4000)
- Compare validation F1 scores between 16D and 32D
- If 32D improves validation F1, it's likely learning better generalizable features

---

## 3. 3D Plotting Implementation

I've added 3D plotting functionality to `patient_aggreagation_encoding_finetune.py`:

### Features:
- **Plots unsupervised embeddings** (after Phase A) for both training set and test patient
- **Plots supervised embeddings** (after Phase B) for both training set and test patient
- **Test patient points are shown in red** with dark red outlines to distinguish from training data
- **Separate plots** for training-only, test-only, and combined views

### Usage:
```bash
python src/patient_aggreagation_encoding_finetune.py \
    --aggregated-npz <path> \
    --test-patient-id 28 \
    --latent-dim 32 \
    --unsup-steps 6000 \
    --sup-steps 4000
```

### Output Files:
- `xcebra_unsupervised/emb_unsup_train_interactive.html` - Training data only
- `xcebra_unsupervised/emb_unsup_test_interactive.html` - Test patient only
- `xcebra_unsupervised/emb_unsup_train_test_interactive.html` - **Combined view (test in red)**
- Same for `xcebra_supervised/emb_sup_*.html`

### What to Look For:
1. **Distribution Overlap**: Do test patient points overlap with training points of the same emotion?
2. **Separation**: Are different emotions well-separated in the embedding space?
3. **Outliers**: Are test patient points far from training distribution? (This would indicate poor generalization)

---

## Summary of Recommendations

### Immediate Next Steps:
1. **Increase latent dimension**: `--latent-dim 32`
2. **Increase steps**: `--unsup-steps 6000, --sup-steps 4000`
3. **Add test patient visualization**: `--test-patient-id 28` (for EC304)
4. **Monitor training curves**: Check if loss is still decreasing
5. **Compare validation F1**: 16D vs 32D to see if higher capacity helps

### Expected Outcomes:
- **If 32D + more steps helps**: Validation F1 should improve, and 3D plots should show test patient points overlapping with training points of the same emotion
- **If still poor**: The issue might be fundamental distribution shift between training patients and EC304, requiring different approaches (domain adaptation, patient-specific calibration, etc.)

### Risk Assessment:
- **Low risk**: Increasing steps (CEBRA is robust to this)
- **Medium risk**: Increasing latent_dim (more capacity, but your dataset is large enough)
- **High reward potential**: Both changes address the core issue (insufficient capacity/learning time for patient-invariant features)


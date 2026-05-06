# Pitfalls Research

| Pitfall | Warning Sign | Prevention | Phase |
|---|---|---|---|
| Class mismatch with assignment | Dataset has coupe/convertible/truck but lacks micro/F1/station wagon | Explicit class mapping and source manifest before training | Dataset phase |
| Dataset imbalance | High validation score but weak minority-class F1 | Per-class counts, weighted loss or sampler, targeted data collection | Dataset/training |
| Test leakage | Same images appear in train and validation or final test assumptions leak into training | Hash/dedup checks and fixed split policy | Dataset phase |
| Oversized model | Best model exceeds 95 MB | Track artifact size in every experiment | Model comparison |
| DINOv3 access failure | HF gated repo cannot be downloaded | Validate access early and keep non-gated baselines | Setup/model phase |
| Overfitting | Training loss drops while validation loss rises | Early stopping, augmentation, dropout/weight decay, more diverse data | Training phase |
| Unexplainable code | Demo questions cannot be answered | Keep modules small, document decisions, avoid unnecessary abstractions | All phases |
| Weak UI demo | UI lacks probability chart or confidence score | Implement Gradio requirements as acceptance tests | UI phase |

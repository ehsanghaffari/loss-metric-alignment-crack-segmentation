# Final v2 protocol

Final controlled demonstration: four datasets (Crack500, DeepCrack, CFD, CrackTree260), six losses, three seeds (0, 1, 2), for 72 runs total.

Training uses U-Net with ImageNet-pretrained ResNet-34, random 448x448 crops, no resizing, AdamW (lr 1e-4, weight decay 1e-4), cosine scheduling, batch size 8, and exactly 100 epochs with early stopping disabled. The best checkpoint is selected by validation dataset-global F1 over thresholds 0.01-0.99.

Loss parameters: Focal gamma=2.0; Focal Tversky alpha=0.3, beta=0.7, exponent=0.75; Dice+clDice weight=0.3 with 5 soft-skeleton iterations; Dice+Boundary uses alpha=max(0.01,1-0.01*epoch).

Offline scoring uses p>=t; fixed 0.5, validation-selected, ODS and OIS thresholds; relaxed-F1 radii {0,2,3,4,5} px; Boundary F1 tolerance 2 px; clDice; 8-connected fragmentation; area and skeleton-length errors; per-image and dataset-global aggregation; and explicit empty-mask rules.

The scorer performs preflight validation of masks, probability maps, shapes, ranges, filenames, and validation-threshold provenance before full scoring.

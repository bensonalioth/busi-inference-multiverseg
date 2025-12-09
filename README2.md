```mermaid
flowchart TD

    %% ============== Data Preparation ==============
    A1[BUSI Dataset: images + masks]
    A2[Diffusion: generate 500 unlabeled images]

    A1 --> A3[Ultrasound Preprocess: CLAHE, log, despeckle]
    A2 --> A4[Gen Dataset: resize + normalize]

    %% ============== Prototype Building ==============
    A3 --> B1[Sample per class: 24 images]
    B1 --> B2[Compute class prototypes]
    B2 --> B3[Merge prototypes into proto_img, proto_msk]

    %% ============== Interactive Prompts ==============
    A3 --> C1[Load batch (img, mask)]
    C1 --> C2[Simulate prompts: pos, neg, box, prev]
    C2 --> C3[Form q5: img + 4 prompt channels]

    %% ============== MultiverSeg ==============
    C3 --> D1[MultiverSeg forward pass]
    B3 --> D1

    D1 --> D2[Predict logits → probs]

    %% ============== Losses ==============
    D2 --> E1[Supervised loss: BCE + Dice]
    A4 --> E2[Entropy loss (every 3 steps)]

    E1 --> F1[Total loss = sup + lambda * unsup]
    E2 --> F1

    %% ============== Optimization ==============
    F1 --> G1[AdamW + Cosine Scheduler]
    G1 --> G2[Update model parameters]

    %% ============== Validation & Saving ==============
    D2 --> H1[Compute Val Dice, Acc, Se, IoU]
    H1 --> H2{Is Val Dice best?}

    H2 -->|Yes| H3[Save gen_best.pt]
    H2 -->|No| H4[Continue training]
```

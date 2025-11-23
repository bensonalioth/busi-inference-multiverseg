```mermaid
flowchart TD

%%========================
%%  DATASET & SAMPLES
%%========================
A1([Dataset Root<br>BUSI_with_GT]) --> A2[Scan Folder<br>build_samples()]
A2 -->|benign/malignant/normal| A3((samples[]))

A3 --> A4[[split_samples()]]
A4 -->|train idx| A5[[BUSIDataset<br>(train)]]
A4 -->|val idx| A6[[BUSIDataset<br>(val)]]
A5 --> D1[DataLoader(train)]
A6 --> D2[DataLoader(val)]

%%========================
%%  PROTOTYPE BUILDING
%%========================
D1 -. small loader .-> P1[[build_class_prototypes()]]
P1 -->|per-class avg img/mask| P2((class_prototypes))
P2 --> P3[[merge_prototypes()]]
P3 -->|proto_img, proto_msk| M1

%%========================
%%  MODEL
%%========================
M1([WrappedMVS<br>MultiverSegNet])
subgraph MVS[MultiverSeg Segmentation Pipeline]
    direction TB
    M1 --> M2[Forward(q5, proto_img, proto_msk)]
    M2 --> M3((Pred Mask))
end

%%========================
%%  TRAINING LOOP
%%========================
D1 --> T1[Training Loop]
T1 --> T2[simulate_interactive_channels()<br>pos/neg/box/prev]
T2 --> T3[Build q5 (5 channels)]
T3 --> M1
M3 --> T4[Compute Loss<br>BCE + Dice]
T4 --> T5[Optimizer Step]
T5 --> T6[CosineAnnealing Scheduler]

%%========================
%%  VALIDATION LOOP
%%========================
D2 --> V1[Validation Loop]
V1 --> M1
M3 --> V2[Dice Metric]

%%========================
%%  CHECKPOINT
%%========================
V2 -->|best dice| C1[[Save best.pt]]


```

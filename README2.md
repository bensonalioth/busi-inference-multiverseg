flowchart TD

%% ========================
%%   SECTION 1: DATA
%% ========================
subgraph DATA[資料準備 Data Preparation]
A1[DATASET_ROOT]:::node --> A2[BUILD_SAMPLES]:::node
A2 --> A3[SAMPLES_LIST]:::node
A3 --> A4[SPLIT_SAMPLES]:::node
end

A4 -->|train| D1[DATASET_TRAIN]:::node
A4 -->|val|   D2[DATASET_VAL]:::node

D1 --> DL1[DATALOADER_TRAIN]:::node
D2 --> DL2[DATALOADER_VAL]:::node

%% 中文解釋
note1[[資料掃描 → 建立 samples list → 分層抽樣成 train/val]]:::note
A1 --- note1

%% ========================
%%   SECTION 2: PROTOTYPE
%% ========================
subgraph PROTO[原型建構 Prototypes]
DL1 -.small loader.-> P1[BUILD_CLASS_PROTOTYPES]:::node
P1 --> P2[CLASS_PROTOTYPES]:::node
P2 --> P3[MERGE_PROTOTYPES]:::node
P3 --> PM[PROTO_IMG_PROTO_MASK]:::node
end

note2[[依類別收集 support 影像與 mask，產生 prototype]]:::note
P1 --- note2

%% ========================
%%   SECTION 3: MODEL
%% ========================
subgraph MODEL[MultiverSeg 模型]
PM --> M1[WRAPPED_MVS_MODEL]:::node
M1 --> M2[FORWARD_PASS]:::node
M2 --> M3[PREDICTED_MASK]:::node
end

note3[[MultiverSeg 前向運算，利用 q5 + prototype 輸出 segmentation mask]]:::note
M1 --- note3

%% ========================
%%   SECTION 4: TRAIN LOOP
%% ========================
subgraph TRAINING[訓練迴圈 Training Loop]
DL1 --> T1[TRAIN_LOOP]:::node
T1 --> T2[SIM_INTERACTIVE_CHANNELS]:::node
T2 --> T3[BUILD_Q5_INPUT]:::node
T3 --> M1
M3 --> T4[COMPUTE_LOSS]:::node
T4 --> T5[OPTIMIZER_STEP]:::node
T5 --> T6[COSINE_LR_SCHEDULER]:::node
end

note4[[產生 pos/neg/box/prev 四種互動提示 → 建 q5 → BCE+Dice → 更新權重]]:::note
T1 --- note4

%% ========================
%%   SECTION 5: VALIDATION
%% ========================
subgraph VAL[驗證 Validation]
DL2 --> V1[VAL_LOOP]:::node
V1 --> M1
M3 --> V2[COMPUTE_DICE]:::node
end

note5[[用 Dice 分數衡量 segmentation 成效]]:::note
V1 --- note5

%% ========================
%%   SECTION 6: CHECKPOINT
%% ========================
V2 -->|best| C1[SAVE_BEST_PT]:::node

note6[[若 Dice 更好 → 儲存 best.pt]]:::note
C1 --- note6


%% ========================
%%   STYLES
%% ========================
classDef node fill:#e7f0ff,stroke:#4a78c2,stroke-width:1px,color:#000
classDef note fill:#fff8dc,stroke:#c2a14a,stroke-width:1px,color:#333

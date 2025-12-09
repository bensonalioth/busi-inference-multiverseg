```mermaid
flowchart TD

    %% ===============================
    %% 原始 BUSI 影像 + Diffusion 生成
    %% ===============================
    A1[原始 BUSI Ultrasound 影像<br/>images + masks] --> A_pre[超音波前處理<br/>(CLAHE, log, despeckle)]
    A2[Stable Diffusion Img2Img<br/>生成 500 張無 GT 影像] --> A3[灰階轉換 + resize]

    A_pre --> B1[建置 Training Samples]
    A3 --> B2[生成影像資料集 (GenImageDataset)]

    %% ===============================
    %% Prototypes
    %% ===============================
    B1 --> C1[以 class 為單位取樣<br/>per_class=24]
    C1 --> C2[平均每類 image + mask<br/>建 class-wise prototypes]
    C2 --> C3[合併為 merged prototype<br/>(proto_img, proto_msk)]

    %% ===============================
    %% Interactive Prompts
    %% ===============================
    B1 --> D1[取 batch (imgs, masks)]
    D1 --> D2[模擬互動提示通道<br/>pos / neg / box / prev]
    D2 --> D3[q5 = imgs + prompts<br/>形成 5-channel query]

    %% ===============================
    %% MultiverSeg
    %% ===============================
    C3 --> E1[MultiverSegNet<br/>(Encoder + Decoder)]
    D3 --> E1

    E1 --> E2[前向推論<br/>logits → probs]

    %% ===============================
    %% Losses
    %% ===============================
    E2 --> F1[Supervised Loss<br/>(BCE + Dice)]
    B2 --> F2[Unsupervised Entropy Loss<br/>每 3 iter 執行一次]
    F1 --> G1[total_loss = sup + λ·unsup]
    F2 --> G1

    %% ===============================
    %% Optimization
    %% ===============================
    G1 --> H1[AdamW + CosineAnnealingLR]
    H1 --> H2[更新參數 (全層微調)]

    %% ===============================
    %% Validation & Saving
    %% ===============================
    E2 --> I1[Validation Dice / Acc / Se / IoU]
    I1 --> I2{ValDice > best?}

    I2 -->|Yes| I3[保存 gen_best.pt]
    I2 -->|No| I4[不更新] 
```

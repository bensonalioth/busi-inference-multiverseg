graph TD
    %% 定義樣式
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef proc fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef storage fill:#e0e0e0,stroke:#424242,stroke-width:2px;

    subgraph Dataset_Preparation [資料準備與分割]
        RawData[("BUSI Dataset<br/>(Benign, Malignant, Normal)")]:::data
        Split[Split Data]:::proc
        TrainDS[Train Set]:::data
        ValDS[Val Set]:::data
        TestDS[Test Set]:::data

        RawData --> Split
        Split --> TrainDS
        Split --> ValDS
        Split --> TestDS
    end

    subgraph Prototype_Construction [Few-Shot 原型建構]
        TrainDS --> SampleProto[取樣: 每類 24 張]:::proc
        SampleProto --> AvgProto[計算平均特徵 (Mean)]:::proc
        AvgProto --> GlobalProto[生成全域 Prototype<br/>(Image + Mask)]:::data
    end

    subgraph Preprocessing_Pipeline [超音波專用前處理]
        %% 這裡詳細列出代碼中的 ultrasound_preprocess 步驟
        RawImg[原始影像] --> ZScore[Z-score Norm]:::proc
        ZScore --> Gamma[Gamma Correction]:::proc
        Gamma --> Log[Log Compression]:::proc
        Log --> CLAHE[CLAHE 增強]:::proc
        CLAHE --> Despeckle[Median Despeckle]:::proc
        Despeckle --> Augs[Augmentations<br/>(Flip, Rotate, Jitter)]:::proc
        Augs --> ReadyImg[Ready Tensor]:::data
    end

    subgraph Training_Loop [Fine-tuning 迴圈]
        PreTrained[("Pre-trained Weights<br/>MultiverSeg_v1")]:::storage
        
        ReadyImg --> ModelInput
        GlobalProto --> ModelInput
        
        ModelInput(輸入: Query x5 + Prototype):::data --> MVS_Net[Wrapped MultiverSeg Net]:::model
        PreTrained -.-> MVS_Net
        
        MVS_Net --> Logits[輸出 Logits]
        Logits --> CalcLoss[計算 Loss<br/>0.5*BCE + 0.5*Dice]:::proc
        CalcLoss --> Backprop[Backprop & Optimizer<br/>(AdamW + AMP)]:::proc
        Backprop --> Scheduler[Scheduler<br/>(CosineAnnealing)]:::proc
    end

    subgraph Validation_Saving [驗證與儲存]
        Scheduler --> EndEpoch{Epoch 結束?}:::decision
        EndEpoch -- No --> ReadyImg
        EndEpoch -- Yes --> Validate[驗證集評估]:::proc
        Validate --> CheckBest{Dice > Best Dice?}:::decision
        
        CheckBest -- Yes --> SaveBest[("儲存 best_mvs.pt")]:::storage
        CheckBest -- No --> CheckDone
        SaveBest --> CheckDone{達到 Max Epochs?}:::decision
        CheckDone -- No --> ReadyImg
    end

    subgraph Final_Testing [測試階段]
        CheckDone -- Yes --> LoadBest[載入 best_mvs.pt]:::proc
        TestDS --> TestPrep[前處理 (無 Augmentation)]:::proc
        TestPrep --> TestInfer[推論 Predict]:::model
        LoadBest --> TestInfer
        TestInfer --> Metrics[計算指標<br/>Acc, Prec, Rec, IoU, Dice]:::data
    end

    %% 連接各個子圖的關係
    TrainDS --> RawImg
    ValDS --> Validate

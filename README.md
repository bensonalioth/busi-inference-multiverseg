```mermaid
flowchart TD

%% ========================
%%  1. 掃描 BUSI 資料集
%% ========================
A1[資料集根目錄<br/>Dataset BUSI_with_GT]
A1 --> A2[掃描資料夾<br/>build_samples]
A2 --> A3[建立 samples 清單<br/>包含 img, mask, label]

%% ========================
%%  2. Train/Val 分割
%% ========================
A3 --> A4[依 label 分層抽樣<br/>split_samples]
A4 -->|Train 索引| A5[建立 BUSIDataset (train)]
A4 -->|Val 索引| A6[建立 BUSIDataset (val)]

A5 --> D1[Train DataLoader<br/>批次資料載入]
A6 --> D2[Val DataLoader]

%% ========================
%%  3. 建立 Prototype
%% ========================
D1 -. 使用小批量 .-> P1[依類別收集 support<br/>build_class_prototypes]
P1 --> P2[每類 prototype<br/>平均影像 + 平均 mask]
P2 --> P3[合併成單一 prototype<br/>merge_prototypes]
P3 --> PM[proto_img 與 proto_msk]

%% ========================
%%  4. MultiverSeg 模型
%% ========================
PM --> M1[WrappedMVS 模型<br/>基於 MultiverSegNet]
subgraph MVS [MultiverSeg Forward 流程]
    M1 --> M2[Forward 計算]
    M2 --> M3[輸出 segmentation mask]
end

%% ========================
%%  5. 訓練流程
%% ========================
D1 --> T1[訓練迴圈<br/>Epoch Loop]

T1 --> T2[生成互動提示通道<br/>simulate_interactive_channels<br/>pos/neg/box/prev]
T2 --> T3[組合成 q5 (五通道輸入)]

T3 --> M1
M3 --> T4[計算 Loss<br/>BCE + Dice]
T4 --> T5[AdamW 更新]
T5 --> T6[使用 Cosine LR 調整學習率]

%% ========================
%%  6. 驗證流程
%% ========================
D2 --> V1[驗證迴圈]
V1 --> M1
M3 --> V2[Dice Metric<br/>評估分割品質]

%% ========================
%%  7. 儲存權重
%% ========================
V2 -->|若優於歷史最佳| C1[儲存 best.pt]


```

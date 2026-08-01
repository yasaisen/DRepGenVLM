# DRepGenVLM 框架實作準則

## 1. 文件定位

本文件描述內層 Python package `DRepGenVLM/` 的整體架構、資料流、跨模組契約與必須維持的不變量。

本文件與下列逐檔準則共同構成目前 DRGVLM 的實作規格：

- [Dataset 實作準則](datasets/multiROI2DxResultDataset.md)
- [Model 實作準則](models/modeling_medGemmaLoRA.md)
- [Loss 實作準則](criterions/DRGVLMLoss.md)
- [Trainer 實作準則](trainer/building_DRGVLM_PPTrainer.md)
- [Inference pipeline 實作準則](pipeline/drgvlmInferencePipeline.md)

本文件採標準 Markdown。所有方程式以純文字 fenced code block 表達，不依賴 LaTeX、MathJax、Mermaid 或特定 renderer extension。

除非文件明確標示為邊界或非保證項目，後續程式修改原則上不得違反此處定義的演算法語義。若有意更改演算法，應在同一次變更中同步更新相關逐檔準則與本文件。

## 2. 框架目標

DRepGenVLM 是 case-level 下游病理報告生成框架。每個 case 可含多個 tissue blocks、stains 與 ROIs；每個 ROI 透過 `DxPair` 宣告它支援的診斷項目 `DxItem`。

框架不把整個 case 的所有 ROI 無差別送給每一個診斷任務，而是建立：

```text
one forward unit = one (case, active DxItem)
```

每個 forward unit 只使用該 DxItem 被分派到的有序 ROI 序列，並生成一段自由文字答案。

核心目標：

- 保留 ROI 與診斷項目的明確關係。
- 使用 frozen multimodal backbone 與 LoRA 進行參數效率微調。
- 以 causal language-model SFT 學習報告文字。
- 在不訓練視覺 adapter 時共享相同 ROI prefix 的視覺計算。
- 同時以一般文字指標與臨床結構指標評估。
- 以可驗證、不可變 snapshot 支援 checkpoint 與精確續訓。
- 對 targetless metadata 執行 deterministic inference 並安全回寫生成報告。

## 3. Package 結構與責任

```text
DRepGenVLM/
  configs/
    DRGVLM_baseConfig.py
    configHandler.py
  datasets/
    multiROI2DxResultDataset.py
    building_datasetHandler.py
    maxROI_sampler.py
  models/
    modeling_medGemmaLoRA.py
    modelBuilder.py
    flashAttentionPatch.py
  criterions/
    DRGVLMLoss.py
  trainer/
    building_DRGVLM_PPTrainer.py
    building_scheduler.py
  evaluator/
    DRGVLMEvaluator.py
  pipeline/
    drgvlmInferencePipeline.py
  common/
    utils.py
    dist_utils.py
    pipeParal_utils.py
    metricsTracker.py
    metricsTracker_v2.py
  projects/
    saved JSON configs
  processors/
    reserved package; currently no concrete implementation
```

責任分界：

| 子系統 | 權威責任 |
| --- | --- |
| Config | 路徑、資料、模型、訓練與 runtime hyperparameters |
| Dataset | metadata 關係驗證、active DxItem ROI 分組、抽樣、影像 I/O |
| Dataset handler/sampler | DataLoader、case batch 與 ROI-assignment budget |
| Model builder | base model/processor 載入、freezing、PP device map、attention backend |
| DRGVLM model | prompt、LoRA、cached/uncached forward、generation |
| Criterion | causal token cross-entropy |
| Trainer | case-balanced backward、optimizer lifecycle、validation、checkpoint |
| Evaluator | prediction completeness、文字 metrics、臨床結構 metrics |
| Inference pipeline | artifact/preflight、targetless prediction、metadata 回寫 |
| Common | seed、distributed helpers、logging、monitoring、通用工具 |

## 4. 跨模組權威來源

同一概念在多個模組間傳遞時，必須遵循以下權威來源：

| 概念 | 權威來源 | 不得替代為 |
| --- | --- | --- |
| DxItem universe 與順序 | metadata top-level `DxItem_list` | model hard-coded list |
| Case identity | metadata `sample_idx` | case list index 或 `case_id` |
| Active DxItem | ROI `DxPair` keys | target 是否存在 |
| 某 DxItem 的模型 ROI | `Case.DxItem_rois[DxItem]` | `Case.rois` 全聯集 |
| SFT 文字 target | `DxResultTxt` | `DxResultCls` |
| 臨床分類 reference | `DxResultCls` | 從 `DxResultTxt` 重新解析 |
| Vision-cache shareability | ordered ROI `global_idx` signature | ROI 集合或 ROI 數量 |
| 訓練 pair 權重 | trainer case-balanced 公式 | criterion 自行加權 |
| Checkpoint truth | immutable snapshot + manifest | 可被覆寫的同名 weights file |
| Inference expected output | active `(sample_idx, DxItem)` 集合 | 生成結果自行決定 |

## 5. 核心資料模型

### 5.1 Metadata hierarchy

```text
metadata
  DxItem_list
  case_list
    case
      sample_idx
      case_id
      structured_report.DxItems      optional only in targetless inference
      tissue_blocks
        stains
          roi_list
            global_idx
            DxPair
            <level_key>
              roi_path
              mpp
              cxcywh
              roi_wh
```

### 5.2 Active relationship

令：

```text
D = ordered top-level DxItem_list
R_c = all ROI records in case c
```

則：

```text
R_(c,d) = [r in metadata order | d is a key of r.DxPair]

d is active for c <=> R_(c,d) is non-empty
```

只有 active DxItem 會產生模型 forward、生成結果與 evaluator expected prediction。

### 5.3 抽樣後資料

```text
R~_(c,d) = sample R_(c,d) independently for each d

U_c = unique union of all R~_(c,d), deduplicated by ROI global_idx
```

`Case.rois` 保存 `U_c`；`Case.DxItem_rois[d]` 保存 `R~_(c,d)` 對應的 ROI object references。

相同實體 ROI 可同時屬於多個 `R~_(c,d)`，但只應開圖與 augmentation 一次。

## 6. Metadata 驗證原則

Training/validation metadata 必須：

- 有非空且無重複的 `DxItem_list`。
- 每個 case 有合法 tissue block/stain/ROI hierarchy。
- 每個 `DxPair` key 已宣告。
- 每個 active DxItem 有 `structured_report` target。
- 每個 target 同時有 `DxResultTxt` 與 `DxResultCls`。
- 每個 case 至少有一個 active DxItem。

Targetless inference 可缺少 `structured_report`，但仍必須：

- 有 `DxItem_list`。
- 有合法 ROI `DxPair` 關係。
- 每個 case 至少有一個 active DxItem。

框架對關係錯誤採 fail-fast，不自動新增 target、不刪除陌生 DxItem、不猜測 ROI assignment。

## 7. ROI sampling 準則

### 7.1 Per-DxItem limit

`max_rois_per_dxitem=K` 是對每個 `(case, DxItem)` 獨立套用，不是整個 case 的唯一 ROI 上限。

### 7.2 Modes

```text
all:
    keep all assignments; ignore K

random_k:
    sample K distinct positions, then restore original relative order

head_k:
    keep first K assignments

tail_k:
    keep last K assignments
```

`K is None`、`K <= 0` 或原 list 長度不大於 K 時保留完整 list。

已確認：`all` 必須忽略 K。

### 7.3 Randomness

Training `random_k` 使用全域 Python RNG，允許跨 epoch 改變，並受 checkpoint RNG state 管理。

Validation/inference `random_k` 使用 pair-specific RNG：

```text
pair_seed = valid_sampling_seed
            + case_index * max(1, number_of_DxItems)
            + DxItem_index
```

同一 metadata ordering 與設定下必須 deterministic。

## 8. ROI assignment resource semantics

框架的 raw/effective ROI count 與 max-ROI batch budget 都以 ROI-to-DxItem assignment 為單位。

```text
one physical ROI assigned to 3 DxItems
=> assignment count is 3
```

這與實際影像 I/O 次數不同，因為實體 ROI 會在 case 內去重。

## 9. DataLoader 與 batch sampler

### 9.1 Dataset handler

`datasetHandler.from_config`：

1. 依 metadata paths 建立 validation dataset。
2. 建立 training dataset。
3. 建立 DataLoaders。
4. 將 train DataLoader 長度寫回 `cfg.num_batchs_per_epoch`。
5. 將 training dataset 的 `DxItem_list` 寫回 config。

dataset collate function 原樣回傳：

```text
List[Case]
```

沒有 tensor padding 或 stacking。

### 9.2 Explicit generators

```text
train generator seed = dataloader_seed
valid generator seed = dataloader_seed + 1
```

workers 不持久化：

```text
persistent_workers = False
```

目的是讓 checkpoint 恢復的 DataLoader generator 能重現 sampler order 與 worker seeds。

### 9.3 Default sampling

- 非 distributed training shuffle：`RandomSampler` 使用 explicit train generator。
- 非 shuffle：`SequentialSampler`。
- distributed：`DistributedSampler`。
- validation 不 shuffle；distributed 時使用不 shuffle 的 `DistributedSampler`。

### 9.4 `MaxROIBatchSampler`

batch 同時受兩個限制：

```text
number of cases <= batch_size
sum of effective ROI assignments <= max_rois_per_batch
```

建構時從 sampler 順序形成 deque。每次找第一個可放入目前 batch 的 case；若前方 case 不合適，暫時 rotate 到 deque 尾端。

已確認的 oversize 規則：

```text
if current batch is empty:
    first selected case may exceed max_rois_per_batch
```

因此單一超額 case 仍獨立形成 batch，避免永遠無法被排程。後續 case 不能再加入超額 batch。

`drop_last=True` 依 case count 是否達到 `batch_size` 決定是否保留最後 batch，不依 ROI budget 是否剛好填滿。

budget 是每個 DataLoader microbatch 的限制，與 gradient accumulation update 的總 ROI 數無關。

## 10. 設定系統

### 10.1 `DRGVLM_baseConfig`

config constructor 依 project mapping 建立四類設定：

- paths。
- dataset/DataLoader。
- model/LoRA/PP/generation。
- trainer/runtime。

主要 DRGVLM project presets 使用 MedGemma 1.5、LoRA、PP 與 shared vision cache。

### 10.2 Local 與 HPC 差異

constructor 的現行行為包括：

- HPC 使用 project preset 的 per-DxItem ROI limit、sampling mode 與 PP GPU 數。
- 非 HPC 強制將 per-DxItem ROI limit 設為 2、mode 設為 `random_k`。
- 非 HPC 若 project 原本有 PP GPU 設定，將 `pp_num_gpus` 設為 1。

saved JSON config 載入時使用檔案內屬性，不重新執行上述 constructor 邏輯。

### 10.3 新舊名稱相容

目前保留：

```text
max_rois_per_case   -> max_rois_per_dxitem
max_rois_per_update -> max_rois_per_batch
```

新舊值同時存在且不同時必須失敗，不能任選其一。

### 10.4 Config save/load

- `to_dict` 保存所有非底線 attributes。
- `save` 以 JSON 寫入設定，explicit save path 優先於 `self.save_path`。
- `load` 使用 `__new__` 後直接恢復 JSON attributes。
- `load` 只補必要的 legacy ROI limit，不重跑 defaults。

### 10.5 `ConfigHandler`

CLI 至少需提供 config、best checkpoint directory 或 latest checkpoint directory之一。

主要模式：

```text
new run:
    config only

keepTrain:
    checkpoint + retrain false or default continuation semantics

reTrain:
    checkpoint + --retrain
```

`--use-this-dir` 只允許 latest checkpoint continuation，用原 checkpoint root 繼續寫入；best checkpoint 不允許原地續寫。

沒有原地續寫時，checkpoint resume 預設建立新的 timestamp output directory。

## 11. Base model 與 processor 建立

### 11.1 Model registry

`modelBuilder` 保存 model name 到 local checkpoint directory/file 的 mapping。DRGVLM 主路徑由 `DownstreamRepGenVLM.from_config` 指定：

```text
project_name = "DRGVLM"
freeze_weight = True
load_visual_processor = True
dtype = bfloat16
use_bidirectional_attention = False
```

### 11.2 Loading

主模型由 Hugging Face Transformers `from_pretrained` 載入：

- local model path。
- config override。
- chosen attention implementation。
- manual PP device map 或 `auto`。
- `low_cpu_mem_usage=True`。
- `trust_remote_code=True`。

processor/tokenizer 的 pad token 在缺少或被要求時設成 EOS token。

### 11.3 Freezing 與 LoRA

base model 載入後先將所有 parameters 設為 `requires_grad=False`。DRGVLM model 再透過 PEFT 注入 LoRA，只有符合 scope 的 adapter parameters 可訓練。

## 12. Pipeline-parallel device map

### 12.1 權威實作

DRGVLM model 載入實際使用：

```text
models.modelBuilder.modelBuilder._build_manual_pp_device_map
```

`common.pipeParal_utils.build_manual_pp_device_map` 是獨立的較舊 helper，不是目前 `DownstreamRepGenVLM.from_config` 的呼叫路徑。修改 PP policy 時不得只改 common helper 而漏掉 model builder 內的權威實作。

### 12.2 Device map 原則

PP map 明確列舉 model submodules，不使用一個包住整棵 model tree 的 catch-all key，以便 Accelerate 在 layer device 邊界放置正確的 hidden-state transfer hooks。

DRGVLM MedGemma 1.5 的目標：

- 視覺 tower、projector、token embeddings、norm 與 tied LM head 依 GPU 數放到指定 home stages。
- transformer text layers 分散到其餘 GPUs。
- project preset 的 8-GPU 路徑可依 `pp_vision_split_index` 將 vision encoder layers 分到 GPU 0/1，text layers 平衡分到 GPU 2 之後。
- unknown model family 或無法取得 layer 數時可回退 `device_map="auto"`。

`pp_vision_split_index` 若使用，必須位於 vision layer range 內。

### 12.3 Model home device

processor 輸入先移到 config `device`，通常為 `cuda:0`。後續 layer-to-layer transfer 由 dispatched model hooks 負責。trainer 不得再對整個 PP model 呼叫 `.to(single_device)`。

## 13. FlashAttention large-head fallback

import `modelBuilder` 時，若可 import `flash_attn_interface`，安裝 idempotent wrapper：

```text
head_dim <= 256:
    call native FlashAttention

head_dim > 256:
    call exact block-query fallback
```

fallback：

- 支援 padded 與 variable-length interfaces。
- 使用 bottom-right aligned causal/window mask。
- float32 計算 scores、softmax 與 gradients。
- 支援 query heads 是 KV heads 整數倍的 GQA repeat。
- query block size 固定為 1024。
- 明確拒絕 attention dropout、softcap、ALiBi、paged/block table、attention probabilities 等未支援選項。

fallback 不得靜默忽略不支援的 attention options，因為那會改變模型語義。

若整個 `flash_attn_interface` 不存在，SDPA 設定可以正常不安裝 patch；其他 patch 安裝錯誤只輸出 warning，實際模型載入仍由所選 attention backend 決定。

## 14. DRGVLM model

### 14.1 Prompt 單位

對每個 `(case, DxItem)`：

```text
for each assigned ROI in order:
    optional image placeholder
    optional BOC when image absent
    optional MPP
    optional cxcywh
    SEP

append:
    "Based on the histological images above, what is the {DxItem}?"
```

training 再加入 assistant target；inference 只加入 generation prompt。

### 14.2 SFT labels

```text
labels = full input_ids
mask user/image/location/template prefix as -100
mask padding as -100
leave assistant answer and ending tokens supervised
```

### 14.3 LoRA scope

base VLM 保持 eval；training 時只開啟 LoRA dropout。PEFT scope 必須驗證 text/vision trainable parameter 數符合 `use_vision_lora`。

### 14.4 Shared vision cache

```text
sig(c,d) = ordered tuple of ROI global_idx
```

只有 signature 完全相同的 DxItems 才能共享 prefix。

prefix 由第一層文字 decoder pre-hook 擷取，並在 `no_grad` 下 detach。最後 SEP 後的問題與答案 suffix 使用 frozen token embeddings重建。

已確認：

```text
shared vision cache and vision LoRA cannot both be enabled
```

### 14.5 Generation

- greedy decoding。
- 同 signature DxItems 可分批 cached generation。
- batched cached 失敗時以 batch size 1 重試。
- sequential cached 仍失敗時，整個 case 回退完整 VLM generation。

## 15. Criterion

每個 pair 的 model logits 與 labels 進入標準 causal shift：

```text
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

cross-entropy：

```text
mean over all non--100 shifted labels of:
    negative log probability of the target token
```

logits 在 CE 前轉為 float32；可套用 config label smoothing。loss 非 finite 時立即失敗。

## 16. Training objective 與 backward

### 16.1 Pair loss

```text
L_(c,d) = token-average causal CE for case c and active DxItem d
```

### 16.2 Case-balanced microbatch objective

```text
B       = cases in one DataLoader microbatch
M       = number of cases in B
D_c     = active DxItems of case c
n_c     = number of active DxItems of c
A       = actual number of microbatches in current accumulation block
```

每個 pair backward 使用：

```text
L_(c,d) / (A * M * n_c)
```

等價於 microbatch contribution：

```text
(1 / A)
* (1 / M)
* sum over c in B of:
    (1 / n_c) * sum over d in D_c of L_(c,d)
```

已確認此設計是 case-balanced，不是將所有 `(case, DxItem)` pairs 全域等權。

### 16.3 Streamed backward

trainer 依 ROI signature group 建立 context，再逐 DxItem：

```text
forward one pair
-> finite check
-> scale
-> backward immediately
-> delete graph references
```

不保留整個 case 所有 pair graphs。

### 16.4 Gradient accumulation

最後不足完整 block 時：

```text
A = actual remaining DataLoader batches
```

optimizer update 前必須檢查：

- 所有 trainable parameters 都有 gradient。
- 至少一個 gradient 非零。
- gradients 全 finite。

optional clipping 後執行 optimizer、parameter/optimizer-state finite checks、scheduler step，再清空 gradient。

## 17. AMP 與數值精度

主要低精度策略：

- base VLM 以 BF16 載入。
- trainer optional CUDA BF16 autocast。
- criterion CE 強制 float32。
- shared-cache path 停用 autocast weight cache。
- shared prefix 與 frozen suffix embeddings 使用 model dtype。
- checkpoint 前檢查 trainable parameters 與 optimizer state finite。

現行 BF16 training 不使用 GradScaler。

## 18. Validation 語義

每個 validation batch：

```text
build case vision contexts once
-> calculate validation loss
-> generate outputs with same contexts
-> evaluator.update
```

model 的 validation loss 是 batch 中所有 pair losses 直接平均；trainer 再對 validation batches 等權平均。

這和 training case-balanced backward 的 weighting 不相同，屬於現行明確行為。

## 19. Evaluator

### 19.1 Prediction completeness

evaluator expected outputs 只包含每個 case 的 active DxItems。

檢查：

- missing prediction。
- empty prediction。
- unexpected case output。
- unexpected DxItem。
- duplicate case。

`strict_prediction_completeness=True` 時，只要有結構錯誤就整批拒絕。lenient mode 下 missing prediction 仍以所有文字 metrics 為 0 納入分母。

### 19.2 文字正規化

文字轉小寫後，以英數 word token pattern 切詞。exact match 比較正規化 token 以單一空白串接的結果。

### 19.3 文字 metrics

每個 DxItem 計算：

- exact match。
- sentence BLEU，採 effective order 與 add-one smoothing。
- ROUGE-L F1，基於 longest common subsequence。
- unigram bag-of-words token precision、recall、F1。

每個 metric 的 macro 值是有實際樣本之 DxItems mean。

Text composite：

```text
mean of available values among:
    macro exact match
    macro BLEU
    macro ROUGE-L
    macro token F1
```

### 19.4 Histologic type

reference 與 prediction 經規則式 canonicalization，明確區分：

- ductal/lobular carcinoma in situ。
- invasive ductal/lobular carcinoma。
- invasive cribriform carcinoma。
- extracellular mucin 等描述。
- unspecified invasion。

過短或 placeholder reference 不納入 type accuracy，並列為 invalid reference。

### 19.5 Nottingham histologic grade

權威 reference 只來自下列 `DxResultCls`：

```text
Histologic_Grade       -> Grade I/II/III
Tubular_formation      -> Score 1/2/3
Nuclear_pleomorphism   -> Score 1/2/3
Mitotic_count          -> Score 1/2/3
```

```text
total_score = tubular + pleomorphism + mitotic

grade_from_total:
    3 to 5 -> grade 1
    6 to 7 -> grade 2
    8 to 9 -> grade 3
```

prediction 從自由文字解析 grade、三個 components 與 total score。若沒有 explicit grade 但有 total，可由 total 補 grade；若兩者都有且衝突，保留 explicit grade 並記錄 conflict。

重要跨模組契約：即使 grade components 本身不是 active forward units，dataset 仍需保留它們的 `DxItem_target_classes`，供 Histologic Grade evaluator 建立權威 reference。

### 19.6 Microcalcification

解析 status：

```text
present
absent
not_identified
unknown
```

`absent` 與 `not_identified` 保留不同臨床語義。另解析 location set，例如 DCIS、invasive carcinoma、non-neoplastic tissue。

metrics 包含 status accuracy、presence accuracy、location set F1 與 parser coverage。

### 19.7 Clinical composite schema v3

Grade task score：

```text
mean of available:
    overall grade accuracy
    tubular accuracy
    pleomorphism accuracy
    mitotic accuracy
    total score accuracy
```

Microcalcification task score：

```text
mean of available:
    status accuracy
    location F1
```

Clinical macro：

```text
mean of available task scores for:
    Histologic Type
    Histologic Grade
    Microcalcification
```

clinical metric schema version 是 checkpoint score 的一部分，不同版本不得直接比較。

### 19.8 評估 artifact

evaluator 可寫出：

```text
eval_results[epoch]_<timestamp>.json
```

包含 epoch、完整 metrics 與 case-level prediction/reference/structured parsing 結果。這些 case records也用於未來 metric schema migration。

## 20. Logging 與監控

### 20.1 `log_print`

只在 main process 輸出，包含時間、class/function context。distributed 初始化後可覆寫 built-in print，使非 master process 安靜。

### 20.2 Seed

`set_seed(seed)` 同時設定：

```text
Python random
NumPy
Torch CPU
Torch CUDA
cuDNN deterministic = True
```

精確 continuation 另由 checkpoint 恢復各 RNG states 與 DataLoader generator state。

### 20.3 Metrics tracker

保存：

- timestamped JSONL records。
- TensorBoard scalars/histograms。
- per-step train loss。
- per-epoch train/validation summaries。
- learning rates，只在 optimizer update 時記錄。

### 20.4 Training monitor

always-on monitoring 依實際 elapsed time 與 step/work delta 計算：

- cases/sec。
- DxPairs/sec。
- ROI assignments/sec。
- time/step。
- global gradient norm。
- 部分 update ratios。
- 各 GPU reserved VRAM。

periodic monitoring 可記錄 layer gradient norms、histograms 與可取得的 attention entropy/heatmap。

## 21. Distributed 邊界

`dist_utils` 支援環境變數或 Slurm 的 process-group 初始化、rank helper、barrier、tensor average 與 bool broadcast。

現行 base config 預設：

```text
distributed = False
world_size = 1
```

主 DRGVLM trainer 是 PP-oriented 單 process trainer。dataset handler 雖可在已初始化 distributed process group 時建立 DistributedSampler，但 trainer 本身沒有 DDP wrapper、跨 rank loss aggregation 或 rank-specific checkpoint orchestration。

因此不得僅將 `cfg.distributed=True` 就假設已取得完整 DDP training semantics；若未來正式支援 DDP，必須同步定義 optimizer、evaluator、checkpoint 與 metrics 的跨 rank 契約。

## 22. Checkpoint 與 resume

### 22.1 Artifact 結構

```text
save_path/
  checkpoint_artifacts/
    immutable snapshot/
      trainer.pth
      adapter/
      manifest.json
  checkpoint_refs/
    best.json
    latest.json
    periodic and metric references
```

snapshot 先在同 filesystem temporary directory 完整建立，再以 `os.replace` 發布。

### 22.2 Integrity

manifest 記錄 snapshot 每個 artifact 的 SHA-256 與大小。trainer 與 inference pipeline載入新版 snapshot 前都必須驗證 hash。

### 22.3 Trainer state

`trainer.pth` 包含：

- epoch/global/optimizer steps。
- optimizer/scheduler。
- best loss、best metric、patience。
- batch count。
- metrics history。
- Python/NumPy/Torch CPU/CUDA RNG。
- train DataLoader generator state。
- resume signature。

LoRA adapter 另存。

### 22.4 References

`best`、`latest`、periodic 與 metric names 都是 atomic JSON references，不是反覆覆寫的 weight files。多個 references 可指向同一真實 snapshot。

### 22.5 Resume signature

exact continuation 必須比對：

- metadata hashes。
- dataset/sampling/budget。
- DataLoader topology 與 batch count。
- accumulation/schedule。
- model/PP/LoRA topology。

不相容時拒絕 continuation。

### 22.6 Modes

```text
keepTrain:
    restore LoRA + optimizer + scheduler + epoch/steps
    + best states + patience + RNG + generator

reTrain:
    restore LoRA only
    reset optimizer/scheduler progress, epoch, steps and best states
```

resume 到新 output directory 時保存 source manifest；不在啟動時重寫歷史 best/latest。

### 22.7 Legacy compatibility

trainer 與 inference 仍可讀取：

```text
[trainer]<weight_filename>
<weight_stem>_lora/
```

但新版 writer 只建立 immutable snapshots 與 references。

## 23. Metric-best checkpoint policy

框架分別追蹤：

- lowest validation loss。
- highest macro exact match。
- highest macro BLEU。
- highest macro token F1。
- highest macro ROUGE-L。
- highest text composite。
- highest clinical macro score。

metric score 只有 finite 值才比較，且必須嚴格高於前值。clinical score 比較前必須確認 schema version 一致。

legacy clinical schema migration 只能在重算後最佳 epoch 仍與現有 adapter epoch 相同時更新 metadata；若最佳 epoch 改變，不得把 adapter重新標示成新最佳。

## 24. Inference pipeline

### 24.1 Artifact 與 config

接受：

- checkpoint reference JSON。
- immutable snapshot directory。
- filename-style request，並可回退 legacy flat checkpoint。

預設選 best reference。saved config 可用存在的 local GPU roots 本地化，再套用 explicit CLI overrides。

### 24.2 Preflight

在載入大型模型前先檢查：

- input/output paths。
- overwrite collisions。
- case identity uniqueness。
- config/metadata DxItem order。
- localized base weight/image roots。
- targetless dataset metadata relationships。
- sampling/generation參數。

GPU availability 與 requested PP GPU count 會在 preflight 成功後、真正建立大型模型前由 `_build_model` 檢查。

### 24.3 Targetless generation

dataset 使用：

```text
split = valid
require_targets = False
```

逐 case 建立 shared contexts 並 greedy generate。模型只載入 criterion-free wrapper 與 LoRA adapter，不恢復 optimizer。

### 24.4 Output validation

predictions 必須和所有 active `(sample_idx, DxItem)` 一一對應。missing、unexpected 或空文字都拒絕。

只有全部 case 成功，且 generated count 等於 preflight expected pair count 時，才原子寫出新 metadata。

輸入檔與輸出檔不得相同。

## 25. 主要執行流程

### 25.1 Training entry

package 外部的 `R27_MVLM_trainDRGVLM_v0.0.py` 依序：

```text
ConfigHandler.get_cfg
-> set global seed
-> optional distributed initialization
-> datasetHandler.from_config
-> DownstreamRepGenVLM.from_config
-> DRGVLM_PPTrainer.from_config
-> save resolved config into run directory
-> trainer.train
-> distributed cleanup
```

### 25.2 Validation entry

`R27_MVLM_validDRGVLM_v0.0.py` 使用相同組裝，但最後呼叫 `trainer.valid`。

### 25.3 Standalone inference

```text
python -m DRepGenVLM.pipeline.drgvlmInferencePipeline
    --checkpoint-dir ...
    --input-metadata ...
    --output-metadata ...
```

也可用 `--preflight-only` 只驗證與顯示預估工作量。

## 26. 跨模組錯誤策略

### 26.1 必須 fail-fast

- metadata relationship 不合法。
- active DxItem 沒有 training target。
- LoRA scope/topology 不符。
- shared cache 與 vision LoRA 同時啟用。
- cache token/mask 邊界無法證明等價。
- loss、gradient、parameter 或 optimizer state 非 finite。
- trainable gradient 缺失或全部為 0。
- exact continuation signature 不相容。
- checkpoint hash 不符。
- evaluator strict completeness 不符。
- inference prediction 集合不完整或含額外結果。

### 26.2 明確 fallback

只有已定義的可恢復情況允許 fallback：

- unknown model PP mapping 可回退 `device_map="auto"`。
- cached generation：batched -> sequential -> full VLM。
- 找不到新版 checkpoint reference 時可嘗試 legacy flat layout。
- FlashAttention interface 不存在時，SDPA 設定可不安裝 patch。

fallback 必須記錄 warning，且不得靜默改變 supervision、active DxItems 或生成結果集合。

### 26.3 Warning-only

- 明顯可疑但仍保留的文字 references。
- explicit warmup 超過建議範圍。
- legacy checkpoint 缺少完整 resume signature/RNG 時，僅能宣告無法保證精確續訓。

## 27. 框架級必須維持的不變量

1. `DxItem_list` 是全框架有序診斷項目 universe。
2. active DxItem 只能由 ROI `DxPair` 定義。
3. ROI 必須 per-DxItem 獨立抽樣。
4. `all` mode 必須忽略 per-DxItem ROI limit。
5. validation/inference `random_k` 必須 deterministic。
6. ROI resource budget 必須維持 assignment-count 語義。
7. 單一 oversize case 必須可獨立形成 batch。
8. 模型必須使用 `DxItem_rois`，不得把 case-wide ROI 聯集送給所有 DxItems。
9. shared prefix 只能在 ordered ROI signature 完全相同時共享。
10. detached shared cache 不得與 vision LoRA 同時使用。
11. frozen backbone 必須保持 frozen/eval，只有 LoRA scope 可訓練。
12. SFT 只能監督 assistant answer，並使用 causal shifted CE。
13. training backward 必須維持 case-balanced weighting。
14. 最後 partial accumulation block 必須按實際大小正規化。
15. `DxResultCls` 是臨床 reference 權威，不得由文字 target重建。
16. checkpoint snapshot 必須 immutable 且可用 SHA-256 驗證。
17. `keepTrain` 與 `reTrain` 的狀態恢復語義必須分離。
18. clinical metric 不得跨 schema version 直接比較。
19. inference 必須精確覆蓋 expected active pair 集合。
20. inference output 必須寫入新檔並採 atomic replace。

## 28. 修改時的合規檢查清單

修改任何 package 程式前後，至少檢查：

### 28.1 Dataset／metadata

- 是否改變 active DxItem 定義？
- 是否改變 ROI ordering 或 signature？
- 是否仍是 per-DxItem sampling？
- assignment counts 是否仍與 sampler 一致？
- target text/class 是否仍分離？

### 28.2 Model／prompt

- cached 與 uncached prompt/labels 是否等價？
- SEP 邊界是否仍唯一可定位？
- processor mask 是否仍與 merged embeddings 對齊？
- LoRA scope 與 shared-cache gradient assumptions 是否仍成立？
- generation output 是否仍只含新 token 或正確切除 prompt？

### 28.3 Loss／trainer

- causal shift 與 `-100` mask 是否未變？
- pair loss 是否仍以 case-balanced 公式縮放？
- scheduler 是否只按 optimizer updates 前進？
- finite checks 是否仍在 optimizer state被污染前執行？

### 28.4 Evaluator

- denominator 是否仍包含 missing predictions？
- grade reference 是否仍使用 authoritative classes？
- clinical score 定義改變時是否提升 schema version？
- metric checkpoint 是否避免跨 schema 比較？

### 28.5 Checkpoint／inference

- snapshot 是否仍 immutable、reference 是否 atomic？
- 新增 artifact 是否納入 hash manifest？
- resume signature 是否涵蓋所有會改變 sample/update sequence 的設定？
- output 是否仍在所有 prediction checks 通過後才寫入？

## 29. 現行框架邊界

- 主要訓練路徑針對 MedGemma 1.5 與目前 Transformers/PEFT 介面設計；其他 model registry 項目不等於全部經 DRGVLM end-to-end 驗證。
- model 沒有在本身強制 context-length truncation；ROI limit 與 batch budget是資源控制，不等於 tokenizer 長度保證。
- PP 依賴 Accelerate device-map hooks，不是顯式 pipeline runtime。
- `processors/` 目前為空 package，processor 由 Hugging Face model path 載入。
- `common.pipeParal_utils` 與 model builder 內有相近 device-map 邏輯，但 DRGVLM 主路徑以 model builder 為準。
- metrics tracker 的 legacy plotting function 仍含非 DRGVLM loss names，主訓練 loop目前不呼叫該 plotting path。
- package 保留一些 DDP helpers，但現行 DRGVLM trainer 的完整語義是 PP/single-process oriented。
- 精確續訓依賴相同 metadata、DataLoader topology、GPU topology、套件行為與 saved RNG states；legacy artifact只能提供較弱保證。

這些邊界是目前實作的真實範圍，不應在沒有相應驗證與規格更新時被宣稱為已支援能力。

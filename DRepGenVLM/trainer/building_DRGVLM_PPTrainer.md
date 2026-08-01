# `building_DRGVLM_PPTrainer.py` 實作準則

## 1. 文件目的與模組角色

本文件描述 `building_DRGVLM_PPTrainer.py` 中 `DRGVLM_PPTrainer` 的現行 optimizer、scheduler、streamed backward、梯度累積、validation、監控、checkpoint、resume 與 metric-best 管理行為，並作為後續修改的實作準則。

此 trainer 的名稱包含 `PP`，現行核心語義是單一 Python process 持有由 Transformers／Accelerate `device_map` 切分到多張 GPU 的模型。trainer 本身不實作 PyTorch pipeline schedule 或 micro-pipeline engine。

主要責任：

- 建立只包含 trainable parameters 的 optimizer。
- 依 optimizer update 數建立 warmup/cosine scheduler。
- 以 `(case, DxItem)` 串流 forward/backward，及早釋放 graph。
- 維持已確認的 case-balanced loss weighting。
- 執行 AMP、finite checks、gradient clipping 與 optimizer step。
- 執行 validation loss、generation 與 evaluator。
- 建立 immutable checkpoint snapshots 與 atomic references。
- 區分 continuation (`keepTrain`) 與 weights-only restart (`reTrain`)。
- 管理 loss-best 與多種 metric-best checkpoints。

## 2. 重要計數器與狀態

| 屬性 | 語義 |
| --- | --- |
| `checkpoint_epoch_idx` | 已載入 checkpoint 完成的 epoch；新訓練為 `None` |
| `global_step` | 已處理的 DataLoader batch 數，不是 optimizer update 數 |
| `optimizer_step` | 實際完成的 optimizer update 數 |
| `best_val_loss` | 歷史最低 validation loss |
| `best_metric_values` | 各 metric-best checkpoint 的 score、epoch 與相關狀態 |
| `patience_counter` | validation loss 未改善的連續 epoch 數 |
| `resume_num_batches_per_epoch` | checkpoint 保存的 epoch batch 數 |
| `_pending_train_generator_state` | 等 DataLoader 建立後才恢復的 generator state |
| `_snapshot_cache_key/path` | 同一訓練狀態重複更新多個 reference 時重用 snapshot |

`device="cuda"` 會正規化為 `"cuda:0"`。

`accumulation_steps` 保證至少為 1；未提供時使用 1。

## 3. Model 存取

`get_model_raw()` 直接回傳 `self.model`。

現行 PP 模式沒有 DDP wrapper 的 `.module` 解包，也不對整體模型呼叫 `.to(device)`；模型 placement 已在 model builder 載入時由 device map 決定。

## 4. Optimizer parameter groups

### 4.1 Trainable parameter 前置條件

建立 optimizer 前收集：

```text
trainable_params = all model parameters where requires_grad == True
```

若數量為 0，必須失敗。對 DRGVLM 正常設定而言，這些參數應為 PEFT LoRA parameters。

### 4.2 Learning-rate scope

`lr_dict` 必須有 `general`。其他非 `None` key 視為 parameter-name substring module scope。

每個 parameter 依以下兩個維度分組：

1. 是否匹配特定 module-name substring。
2. 是否匹配 no-decay substring。

no-decay patterns：

```text
bias
LayerNorm.weight
BatchNorm.weight
norm.weight
```

一般 scope：

```text
lr = lr_dict["general"]
```

特定 module scope：

```text
lr = lr_dict[module_name]
```

no-decay parameters 的 `weight_decay=0.0`，其他 parameters 使用設定的 weight decay。

### 4.3 AdamW

```text
optimizer = AdamW(
    parameter_groups,
    lr=general_lr,
    betas=(0.9, 0.999) by default
)
```

optimizer 不包含 `requires_grad=False` 的 frozen base parameters。

## 5. Scheduler

### 5.1 Step 單位

```text
updates_per_epoch = ceil(num_batches_per_epoch / accumulation_steps)
total_steps = updates_per_epoch * num_epochs
```

scheduler 每次 `optimizer.step()` 後才 step；不隨每個 DataLoader batch step。

### 5.2 Warmup 決策

自動建議值：

```text
automatic_warmup_steps = int(
    min(warmup_ratio * total_steps, max_warmup_steps)
)
```

若 `cfg.warmup_steps is None`，採自動值。

若使用者明確設定：

- 轉成 `int`。
- 負值必須失敗。
- 即使高於自動建議、`max_warmup_steps` 或 `total_steps`，只警告並保留明確設定。

### 5.3 Scheduler 形狀

```text
warmup_steps == 0:
    CosineAnnealingLR(T_max=total_steps)

0 < warmup_steps < total_steps:
    LinearLR(start_factor=0.01, total_iters=warmup_steps)
    then CosineAnnealingLR(T_max=total_steps-warmup_steps)

warmup_steps >= total_steps:
    LinearLR(start_factor=0.01, total_iters=total_steps)
```

## 6. Metrics 與 monitor 初始化

`init_metrics` 建立：

- `MetricsTracker(save_path)`：JSONL、TensorBoard、train/validation loss history。
- `TrainingMonitor`：throughput、gradient norm、VRAM 與週期性梯度統計。

`global_step` 在初始化 metrics 時設為 0。

`init_evaluator` 掛載 `DRGVLMEvaluator`。

## 7. 數值與梯度防護

### 7.1 Loss finite check

`loss_dict` 中每個 tensor 都必須 finite。

### 7.2 Parameter finite check

所有 `requires_grad=True` parameters 都必須 finite。

### 7.3 Gradient finite check

所有存在的 trainable gradients 都必須 finite。

### 7.4 Gradient presence check

每次 optimizer update 前必須同時滿足：

```text
at least one trainable parameter exists
every trainable parameter has grad != None
at least one trainable gradient contains a non-zero element
```

缺少任一 trainable parameter 的 gradient 即失敗，不接受部分 adapter 因拓撲錯誤而未連入 autograd graph。

所有 trainable gradients 全為精確 0 也必須失敗。

### 7.5 Optimizer state finite check

optimizer state 中每一個 tensor，例如 Adam moments，都必須 finite。檢查時會將 parameter object 映射回 trainable parameter name 供錯誤診斷。

### 7.6 診斷摘要

非有限 tensor 的錯誤包含：

- shape。
- dtype。
- 非有限元素數。
- finite min/max，或全非有限標記。
- epoch、batch、case、DxItem 等 caller context。

任何非有限狀態都不得以 skip batch、清零 loss 或自動降低 learning rate 方式吞掉。

## 8. Gradient accumulation block

### 8.1 實際 block 大小

對 DataLoader batch index `b`：

```text
block_start = floor(b / accumulation_steps) * accumulation_steps

actual_block_size = min(
    accumulation_steps,
    num_batches - block_start
)
```

並保證至少為 1。

因此 epoch 最後不足完整 accumulation block 時，分母使用實際剩餘 batch 數，不會因固定除以 `accumulation_steps` 而縮小最後一次 update。

### 8.2 Optimizer step 邊界

```text
is_update_boundary =
    ((batch_idx + 1) % accumulation_steps == 0)
    or is_last_batch
```

epoch 開始先 `zero_grad(set_to_none=True)`，每次 optimizer update 後再清空。

## 9. 訓練 loss 加權準則

### 9.1 定義

對一個 DataLoader microbatch：

```text
B                  該 microbatch 的 case list
M                  len(B)，至少視為 1
D_c                case c 的 active DxItems
n_c                len(D_c)
A                  該 accumulation block 的實際 microbatch 數
L_(c,d)            case c、DxItem d 的 token-average CE
```

每個 pair 執行 backward 的 scalar 為：

```text
scaled_pair_loss = L_(c,d) / (A * M * n_c)
```

所以單一 microbatch 對 update objective 的貢獻為：

```text
microbatch_objective =
    (1 / A)
    * (1 / M)
    * sum over c in B of:
        (1 / n_c) * sum over d in D_c of L_(c,d)
```

一個 accumulation block 的梯度是各 microbatch objective contribution 之和。

### 9.2 已確認的語義

此 weighting 是 case-balanced：

- 每個 microbatch 先讓 case 等權。
- 每個 case 內再讓 active DxItems 等權。
- 不因某 case 有更多 DxItems 而提高該 case 的總權重。
- 不因某 DxItem 有更多 ROI 或更多 answer tokens 而額外提高 pair 權重；pair 內 CE 已先對有效 token 平均。
- 不把整個 microbatch 的所有 `(case, DxItem)` pairs 視為全域等權集合。

後續修改不得改成 pair-global weighting，除非實作準則同步修訂。

## 10. `_backward_train_batch` 流程

### 步驟 1：正規化 batch

輸入轉為 `List[Case]`；list、tuple 以外的單一物件包成 list。

### 步驟 2：逐 case 取得 active DxItems

若 case 沒有 active DxItem，必須失敗。

同時計算：

```text
unique_rois       = len(case.rois)
roi_assignments   = sum_d len(Case.DxItem_rois[d])
DxItems           = n_c
```

### 步驟 3：按 ROI signature 分組

使用 model 的 `_group_dxitems_by_roi_signature`。每個 group 分別建立 shared context，避免不同 ROI 序列誤用同一 prefix。

### 步驟 4：建立 context

在 autocast context 中呼叫：

```text
prepare_case_loss_context(case, dx_items=group_dx_items)
```

shared cache 關閉時回傳 `None`。

### 步驟 5：逐 DxItem forward/backward

每個 pair：

1. 取出自己的 context。
2. 在 autocast 下呼叫 `calculate_dxitem_loss`。
3. 驗證所有 loss finite。
4. 依 case-balanced 公式縮放。
5. 立即呼叫 `backward()`。
6. 只保存 detached Python float 作為 logging buffer。
7. 刪除 pair graph references。

這種 streamed backward 避免先保存整個 case 所有 DxItem graphs 再一次 backward。

### 步驟 6：回傳 logging loss

回傳值中的 loss 不是用於 backward，而是：

```text
先平均每個 case 的 pair loss
再平均 batch 中各 case 的平均 loss
```

另回傳整個 batch 的 active DxItem pair 數。

## 11. `epoch_train`

1. 將 model 設為 training wrapper mode；model 內部仍保持 frozen VLM eval、LoRA dropout train。
2. reset monitor phase 為 `train`。
3. optimizer gradients 清空。
4. 逐 DataLoader batch 執行 `_backward_train_batch`。
5. 每個 DataLoader batch 後：
   - `global_step += 1`。
   - 記錄 cases、DxPairs 與 ROI assignments throughput。
   - 記錄 periodic diagnostics。
   - 將該 batch 的 logging loss 加入 epoch buffer。
6. update boundary 時依序：
   - 檢查 gradients 都存在。
   - 檢查 gradients finite。
   - optional gradient clipping。
   - 記住 step 前 learning rate。
   - `optimizer.step()`。
   - 檢查 trainable parameters finite。
   - 檢查 optimizer state finite。
   - `scheduler.step()`。
   - `optimizer_step += 1`。
   - 清空 gradients。
7. 每個 DataLoader batch 都寫入 train loss metrics；learning rate 只在真正 optimizer update 時記錄。

epoch 回傳：

```text
mean(logging loss of every DataLoader batch)
```

所以 epoch loss 對 DataLoader batches 等權，不依 batch 中 case 數或 pair 數重新加權。

## 12. AMP 行為

autocast 設定：

```text
device_type = "cuda"
dtype = torch.bfloat16
enabled = self.amp
```

shared vision cache 啟用時：

```text
cache_enabled = False
```

原因是防止 `no_grad` prefix pass 中的 detached FP32-to-BF16 LoRA weight cast 被 autocast weight cache 重用於後續 trainable forward。

本 trainer 沒有使用 GradScaler；BF16 路徑直接 backward。

## 13. Gradient clipping

只有下列條件成立時執行：

```text
gradient_clip_norm is not None
and
gradient_clip_norm > 0
```

使用：

```text
torch.nn.utils.clip_grad_norm_(
    all trainable parameters,
    max_norm=gradient_clip_norm,
    error_if_nonfinite=True
)
```

clip 前後都檢查 gradients finite，回傳的 total grad norm 也必須 finite。

## 14. Validation 與 evaluation

### 14.1 `epoch_validEval`

整個函式受 `torch.no_grad()` 保護。

每個 validation batch：

1. 將 model 設為 eval。
2. 對每個 case 預先建立一次 `case_contexts`。
3. 在 optional BF16 autocast 下呼叫 `calculate_loss`。
4. 保存 detached validation loss。
5. 若 evaluator 存在，使用同一批 contexts 呼叫 `generate_outputs`。
6. 將 cases 與 generated outputs 傳給 evaluator。

同一 case contexts 同時供 loss 與 generation 使用，避免同一 validation batch 重複建立視覺 prefix。

### 14.2 Validation loss 彙總

model 的 `calculate_loss` 先將該 batch 所有 pair losses 直接平均。trainer 再對所有 validation batch loss 等權平均：

```text
val_loss = mean(batch_pair_mean_loss for every validation batch)
```

因此 variable-size validation batches 不會再按 pair 數做全 epoch 加權。

### 14.3 Evaluator

epoch 結束時 evaluator：

- 計算文字與臨床 metrics。
- 可保存 case-level `eval_results` JSON。
- 回傳 flatten metrics 供 TensorBoard 與 metric checkpoint 選擇。
- evaluation 後重設自己的暫存狀態。

平均 validation loss 會以單一 val metrics update 寫入 tracker。

## 15. Validation-best 與 early stopping

`_record_validation_result` 嚴格使用：

```text
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    new_best = True
else:
    patience_counter += 1
    new_best = False
```

相等不視為改善。

若 `patience_counter >= early_stop_patience`，在該 epoch 所有必要 checkpoint 與 metrics summary 完成後停止。

若沒有 validation loader，`best_val_loss` 與 patience 不更新。

validation loader 存在且 `val_loss == 0.0` 時：

1. 保存 `error_quick_checkpoint_epoch_<epoch>.pth` reference。
2. 拋出 `ValueError`。

## 16. Epoch-level training orchestration

### 16.1 起始 epoch

```text
checkpoint_epoch_idx = loaded value, otherwise -1
first_epoch = checkpoint_epoch_idx + 1
```

`reTrain` 載入時 checkpoint epoch 會被重設為 `-1`，因此從 epoch 0 開始。

### 16.2 每個 epoch 順序

```text
train
-> optional validation/evaluation
-> reject zero validation loss
-> update best validation loss/patience
-> update metric-best states and references
-> save loss-best if improved
-> write metrics epoch summary
-> periodic checkpoint when epoch_idx % save_freq == 0
-> always update latest reference
-> optional early stop
```

訓練結束關閉 metrics file handle。

## 17. Checkpoint artifact 模型

### 17.1 新版 schema 的目錄角色

```text
save_path/
  checkpoint_artifacts/
    epoch_<...>_step_<...>_opt_<...>_<uuid>/
      trainer.pth
      adapter/
      manifest.json
  checkpoint_refs/
    best.json
    latest.json
    checkpoint_epoch_<n>.json
    best_<metric>.json
```

`checkpoint_artifacts` 中的 snapshot 建立完成後視為 immutable。`checkpoint_refs` 是可原子更新、指向 snapshot 的小型 JSON pointer。

### 17.2 Snapshot ID

包含：

```text
epoch index
global step
optimizer step
8-character UUID suffix
```

先在 artifact root 內建立 temporary directory，完整寫入後以 `os.replace` 發布為 final snapshot。

### 17.3 `trainer.pth` payload

schema version 2 保存：

- epoch、global step、optimizer step。
- best validation loss。
- best metric states。
- patience。
- batches per epoch。
- optimizer state。
- scheduler state。
- metrics train loss history。
- Python、NumPy、Torch CPU 與所有 CUDA RNG states。
- train DataLoader generator state。
- resume signature。

LoRA adapter 另存於 `adapter/`，不嵌入 trainer state。

### 17.4 Manifest 與 integrity

snapshot 內除 `manifest.json` 外的每個檔案都記錄：

- 相對路徑。
- SHA-256。
- byte size。

載入新版 snapshot 或 reference 時，必須逐檔驗證存在性與 SHA-256。驗證失敗不得繼續載入。

### 17.5 Reference atomic write

reference JSON 與其他 trainer JSON manifest 使用：

```text
mkstemp in destination directory
-> json.dump
-> flush
-> fsync
-> os.replace
```

失敗時移除 temporary file。

### 17.6 Snapshot reuse

同一 epoch 訓練狀態若連續更新 best metric、best loss、periodic 與 latest references，可依 state cache key 重用同一 immutable snapshot。

cache key 包含：

- epoch。
- global/optimizer steps。
- best validation loss。
- best metric states。
- patience。
- metrics loss record 數。

它的目的只是在同一狀態下讓多個 reference 指向同一 snapshot。

## 18. Checkpoint request 解析

載入 path 依序支援：

### 18.1 Reference JSON

JSON 必須含 `snapshot`。解析 checkpoint root、驗證 snapshot，再回傳：

```text
snapshot/trainer.pth
snapshot/adapter
checkpoint_root
```

### 18.2 Immutable snapshot directory

directory 必須直接含 `manifest.json`，並通過完整性驗證。

### 18.3 Weight-filename-style request

例如 `save_path/best_model.pth`：

1. 先映射成 `checkpoint_refs/best.json`。
2. 若 reference 存在，遞迴依 reference 載入。
3. 否則嘗試 legacy flat checkpoint。

### 18.4 Legacy flat checkpoint

```text
[trainer]<weight_filename>
<weight_filename_without_.pth>_lora/
```

新版只保留 reader 相容，不再用 flat layout 寫入。

## 19. 初始 artifact 與 resume source

### 19.1 全新訓練

`checkpoint_epoch_idx < 0` 時，在正式 epoch 前保存 `initial_model.pth` reference，snapshot epoch 為 `-1`。

### 19.2 Resume

resume 不會在啟動時重寫 best/latest，避免尚未完成新 epoch 就錯誤標示新 artifact。

若新的 `save_path` 與 checkpoint source directory 不同，建立：

```text
resume_source_checkpoints.json
```

記錄歷史 best loss、metric checkpoints 與可能的 upstream resume manifest。這只是參照來源，不複製或重新標記歷史 adapter。

## 20. Resume signature

### 20.1 Static signature

schema version 2 包含：

- train/valid metadata SHA-256 與檔案大小。
- batch size、DataLoader seed。
- image/location input flags 與 level key。
- per-DxItem ROI limit 與 sampling mode。
- max-ROI sampler 設定與 batch ROI budget。
- `DxItem_list`。
- accumulation、total steps、warmup steps。
- model name、training mode、PP topology。
- LoRA rank、alpha、dropout、targets、vision-LoRA flag。

### 20.2 Runtime signature

DataLoader 建立後加入：

- dataset length。
- DataLoader batch count。
- sampler type。
- batch sampler type。

### 20.3 Continuation 相容性

若是非 `reTrain` 且 loaded epoch 不小於 0：

- 新舊 signature 必須遞迴完全一致。
- 差異會列出完整 key path 與 checkpoint/current values。
- 最多在錯誤訊息展示前 40 個差異。
- legacy checkpoint 沒有 signature 時只警告，改用可取得的 batch-count check。
- checkpoint 的 batches-per-epoch 若與目前 DataLoader 長度不同，必須失敗。

不得在資料、sampling、optimization 或模型拓撲改變時假裝是 exact continuation。

## 21. RNG 與 DataLoader continuation

保存／恢復：

```text
Python random state
NumPy RNG state
Torch CPU RNG state
all CUDA RNG states
train DataLoader torch.Generator state
```

CUDA checkpoint 若包含 RNG state：

- 目前必須有 CUDA。
- 可見 CUDA device count 必須與 checkpoint 相同。

DataLoader generator state 在 `load_checkpoint` 時先暫存，等 `train_dataloader` 傳入 `_prepare_resume_runtime` 後才套用。

若 checkpoint 有 generator state、目前 DataLoader 卻沒有 explicit generator，必須失敗。

這些機制共同維持 sampler order、worker seeds、augmentation 與 train `random_k` 的 continuation。

## 22. `keepTrain` 與 `reTrain`

### 22.1 `keepTrain`／一般 continuation

恢復：

- LoRA adapter。
- optimizer state。
- scheduler state。
- completed epoch。
- global step 與 optimizer step。
- best validation loss。
- best metric states。
- patience。
- RNG 與 DataLoader generator state。
- resume signature compatibility。

缺少 optimizer 或 scheduler state 時必須要求改用 `reTrain`，不能部分 continuation。

### 22.2 `reTrain`

只重用 LoRA weights，並重設：

```text
checkpoint_epoch_idx = -1
global_step = 0
optimizer_step = 0
best_val_loss = infinity
best_metric_values = {}
patience_counter = 0
```

optimizer 與 scheduler 使用 `from_config` 剛建立的新狀態，不載入 checkpoint state。

## 23. Metric-best checkpoints

### 23.1 直接 metric mapping

| Metric | Checkpoint request filename |
| --- | --- |
| `macro_exact_match` | `best_macro_exact_match.pth` |
| `macro_bleu` | `best_macro_bleu.pth` |
| `macro_token_f1` | `best_macro_token_f1.pth` |
| `macro_rouge_l` | `best_macro_rouge_l.pth` |
| `text_composite` | `best_text_composite.pth` |
| `clinical_macro_score` | `best_clinical_composite.pth` |

只有可轉成 finite float 的 score 才參與比較。

### 23.2 改善條件

```text
new score > previous score
```

相等不更新。

新狀態記錄 score、epoch、global step、validation loss 與所有可轉成 finite float 的 metrics。

改善的 metric 各自更新 checkpoint reference；最後以 atomic JSON 寫出：

```text
best_metric_checkpoints.json
```

### 23.3 Clinical metric schema

clinical checkpoint 必須有 `clinical_metric_schema_version`。不同 schema version 的 score 不得直接比較。

載入 legacy clinical state 時：

1. 從 checkpoint directory 找所有 `eval_results*.json`。
2. 用目前 evaluator schema 重算 clinical score。
3. 找出重算後最佳 epoch。
4. 若最佳 epoch 與既有 adapter 所屬 epoch 不同，必須失敗，不能把舊 adapter 錯誤重新標籤。
5. 若 epoch 相同，更新 score、schema version、metrics 並保留 legacy score。

## 24. `from_config` 組裝順序

1. 建立 trainer 基本狀態。
2. 計算 `total_steps`。
3. 決定 warmup steps。
4. 建立 optimizer 與 scheduler。
5. 建立 metrics tracker 與 monitor。
6. 建立 `DRGVLMEvaluator`。
7. 建立 static resume signature。
8. 若提供 checkpoint path，最後載入 checkpoint。

這個順序讓 continuation 能將 optimizer/scheduler state 載入已存在的實例；`reTrain` 則保留新建狀態。

## 25. 必須維持的不變量

1. 訓練 backward 必須維持 case-balanced，而非 pair-global weighting。
2. 最後不足完整 accumulation block 時必須用實際 block size 正規化。
3. 每個 `(case, DxItem)` 必須立即 backward 並釋放 graph，不應聚合整個 case 的大型 graphs。
4. scheduler 只能在 optimizer update 後 step。
5. `global_step` 與 `optimizer_step` 必須保持不同語義。
6. optimizer update 前必須驗證所有 trainable gradients 存在、finite，且不全為 0。
7. optimizer step 後必須驗證 trainable parameters 與 optimizer state finite。
8. shared vision cache 路徑必須關閉 autocast weight cache。
9. validation loss 與 generation 應重用同一批 case contexts。
10. snapshot 必須 immutable；best/latest/metric 名稱只能是 atomic references。
11. 新版 snapshot 載入前必須做 SHA-256 integrity verification。
12. `keepTrain` 必須恢復完整狀態並檢查 signature；`reTrain` 只能重用 LoRA weights。
13. resume 不得在完成新 epoch 前改寫歷史 best/latest。
14. clinical scores 不得跨 metric schema version 直接比較。
15. 保存 checkpoint 前必須檢查 parameters 與 optimizer state finite。

## 26. 現行邊界與非保證項目

- trainer 的 PP 是依賴模型 `device_map` 的單 process 多 GPU，不是 trainer 自己的 pipeline engine。
- epoch train/validation loss 都是 batch-level mean 的再平均；variable-size batches 不做全 epoch sample/pair 加權。
- `plot_freq` 目前保存但未在主 training loop 觸發 `plot_metrics`。
- checkpoint payload 會保存 metrics loss history，但現行 `load_checkpoint` 沒有把該 history 重新注入新 `MetricsTracker`。
- `MaxROIBatchSampler.set_epoch` 未由此 trainer 的 epoch loop 顯式呼叫；非 DDP `RandomSampler` 仍由 generator state控制每次 iteration 的順序。
- `nullcontext()` 包住 `_backward_train_batch`，目前不增加同步或 no-sync 語義。
- checkpoint snapshot cache key 是訓練狀態摘要，不直接 hash model parameter bytes；它用於同一狀態下多 reference 共用，不應跨訓練更新重用。

這些是現行行為邊界。任何要改變訓練統計定義、續訓精確性或 checkpoint truthfulness 的修改，都必須同步更新本準則。

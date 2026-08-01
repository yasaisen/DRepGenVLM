# `drgvlmInferencePipeline.py` 實作準則

## 1. 文件目的與模組角色

本文件描述 `drgvlmInferencePipeline.py` 的 checkpoint 解析、preflight、設定本地化、targetless dataset 建立、逐 case 生成、prediction 驗證與 metadata 原子寫入行為，並作為後續修改此 inference pipeline 的實作準則。

此 pipeline 的目標是：

```text
已訓練的 DRGVLM checkpoint
+ 含 ROI DxPair 的輸入 metadata
-> 為每個 active DxItem 生成文字
-> 寫入每個 case 的 generated_report
```

此 pipeline 不會：

- 修改輸入 metadata 檔案本身。
- 為 inactive DxItem 生成答案。
- 由 `structured_report` 推論 active DxItems。
- 執行 loss 或 evaluator。
- 自動建立不存在的輸出目錄。
- 自動下載 base model weights。

## 2. 主要常數與輸出欄位

```text
GENERATED_REPORT_KEY = "generated_report"

SUPPORTED_ROI_SAMPLING_MODES =
    "all", "random_k", "tail_k", "head_k"
```

每個成功處理的 case 最終增加：

```json
{
  "generated_report": {
    "Histologic_Type": "generated text",
    "Histologic_Grade": "generated text"
  }
}
```

只包含該 case 由 ROI `DxPair` 宣告的 active DxItems，並維持頂層 `DxItem_list` 順序。

## 3. `PipelineArtifacts`

immutable dataclass 欄位：

| 欄位 | 語義 |
| --- | --- |
| `config_path` | 實際使用的 saved DRGVLM config |
| `checkpoint_path` | 使用者 request 或解析後 reference path |
| `trainer_checkpoint_path` | snapshot/legacy 中的 trainer state path |
| `adapter_path` | LoRA adapter directory |
| `checkpoint_root` | checkpoint 共同根目錄 |
| `checkpoint_epoch` | 新版 manifest 可取得的 epoch；legacy 可能為 `None` |

inference 實際載入模型時只需要 config 與 adapter；trainer checkpoint path 仍在 preflight 中驗證及回報，確保 request 指向完整 checkpoint artifact。

## 4. JSON 讀寫契約

### 4.1 `_load_json`

- 使用 UTF-8。
- root 必須是 JSON object。
- 非 object root 必須拋出 `ValueError`。

### 4.2 `_atomic_dump_json`

輸出目錄必須事先存在，pipeline 不自行建立。

寫入順序：

```text
mkstemp in output directory
-> json.dump(indent=4, ensure_ascii=False)
-> append newline
-> flush
-> fsync
-> os.replace(temp, final)
```

任何例外發生時，若 temporary file 尚存在就移除。

此機制保證成功時 final path 一次替換，避免留下部分寫入的 JSON。

## 5. Checkpoint snapshot 完整性

### 5.1 SHA-256

檔案以 1 MiB chunks 讀取並計算 SHA-256。

### 5.2 `_verify_snapshot`

snapshot directory 必須含：

```text
manifest.json
trainer.pth
adapter/
```

manifest 的 `files` 必須是非空 mapping。每個 file record 必須：

- relative path 是字串。
- record 是 mapping。
- 指定檔案存在。
- 有非空 `sha256`。
- 實際 SHA-256 與 manifest 完全一致。

任一條件失敗都不得繼續 inference。

## 6. Checkpoint request 解析

### 6.1 Reference 名稱

```text
best_model.pth   -> best
latest_model.pth -> latest
other_name.pth   -> other_name
```

### 6.2 JSON reference

若 request 是 `.json` file：

1. JSON 必須有非空 `snapshot`。
2. 若 reference 位於 `checkpoint_refs/`，其 parent 的 parent 是 checkpoint root。
3. `snapshot` 相對 checkpoint root 解析。
4. 驗證 snapshot manifest 與所有 hashes。
5. 驗證 `trainer.pth` file 與 `adapter/` directory 存在。
6. epoch 優先取 manifest `epoch_idx`，其次取 reference `epoch_idx`。

### 6.3 Immutable snapshot directory

若 request 是 directory：

- 必須直接包含 `manifest.json`。
- 必須通過完整 hash verification。
- checkpoint root 視為 snapshot parent 的 parent，即一般的 `save_path`。

普通 checkpoint root directory 不能直接當作 snapshot request。

### 6.4 Weight-filename-style request

例如：

```text
/checkpoint/root/best_model.pth
```

先嘗試：

```text
/checkpoint/root/checkpoint_refs/best.json
```

若存在，依 reference 重新解析。

### 6.5 Legacy flat checkpoint

若沒有 reference，嘗試：

```text
/checkpoint/root/[trainer]best_model.pth
/checkpoint/root/best_model_lora/
```

兩者都必須存在。legacy checkpoint 沒有 snapshot hash verification，`checkpoint_epoch` 回傳 `None`。

### 6.6 預設 checkpoint

`resolve_checkpoint_artifacts` 的 `checkpoint_path=None` 時：

```text
if checkpoint_refs/best.json exists:
    use it
else:
    request best_model.pth and allow legacy resolution
```

`checkpoint_dir` 與 config file 必須存在。config 預設為：

```text
checkpoint_dir/config.json
```

## 7. Active DxItem 的 inference 定義

`_case_active_dxitems` 遍歷：

```text
case.tissue_blocks
-> block.stains
-> stain.roi_list
-> roi.DxPair keys
```

先收集所有出現的 key，再依頂層 `declared_dxitems` 順序回傳交集。

因此：

```text
d is active for inference
<=>
d is declared in top-level DxItem_list
and
at least one ROI DxPair contains d
```

`structured_report` 是否有該 DxItem 不影響 active 定義。

## 8. Case identity 驗證

每個 `case_list` 元素必須：

- 是 object。
- 有 `sample_idx`。
- `sample_idx` 可 hash。
- `sample_idx` 在整份 metadata 唯一。
- 有 `case_id`。

`sample_idx` 是 prediction mapping 與 metadata 回寫的唯一 join key。pipeline 不使用 case list index 或 `case_id` 代替。

## 9. Existing report collision

當 `overwrite=False` 時，任何 case 已有 truthy `generated_report` 都會使整次操作失敗。

```text
existing generated_report is truthy
and overwrite is False
=> FileExistsError
```

空 object、空字串或其他 falsy 值不觸發此檢查。

當 `overwrite=True` 時，新結果會完整取代該 case 的 `generated_report` mapping。

## 10. Prediction 套用契約

`apply_generated_reports` 接受：

```text
metadata: mutable JSON object
predictions:
    sample_idx -> DxItem -> prediction object
```

對每個 case：

1. 依 `DxPair` 計算 expected active DxItems。
2. 必須存在該 `sample_idx` 的 prediction mapping。
3. prediction DxItem keys 必須與 expected keys 完全相等。
4. 每個 prediction 必須是 mapping。
5. `pred_txt` 必須是非空、非純空白字串。
6. 寫入 `pred_txt.strip()`。

集合一致條件：

```text
set(case_predictions) == set(expected_active_dxitems)
```

所有 metadata cases 完成後，predictions 不得再含未知 `sample_idx`。

回傳值是成功寫入的 active `(case, DxItem)` 數量，不是 case 數。

`apply_generated_reports` 先直接修改傳入的 in-memory metadata；只有所有檢查成功後，`run` 才原子寫入 output file。

## 11. Pipeline 建構參數

必要參數：

- `checkpoint_dir`
- `input_metadata_path`

`output_metadata_path` 可在只呼叫 `preflight()` 時省略；完整 `run()` 必須提供。

可選 overrides：

- config path。
- checkpoint path。
- overwrite。
- image root。
- base model weight root。
- PP GPU 數。
- home device。
- per-DxItem ROI limit。
- sampling mode／seed。
- maximum generated tokens。
- cached prompt batch size。
- AMP。

所有 filesystem paths 在 pipeline state 中轉為 absolute path；`image_path` 與 `weight_path` overrides 在套用 config 時才轉為 absolute。

建構時立即解析 checkpoint artifacts，但不載入 metadata、dataset 或模型。

## 12. Saved config 本地化與 override 優先序

### 12.1 載入

使用 `DRGVLM_baseConfig.load(config_path)`，直接還原 saved JSON attributes，不重新執行 base config constructor defaults。

### 12.2 Local root fallback

依序檢查下列配對：

| Runtime attribute | Local candidate attribute |
| --- | --- |
| `image_path` | `localGPU_image_path` |
| `weight_path` | `localGPU_weight_path` |
| `metadata_path` | `localGPU_metadata_path` |
| `root_path` | `localGPU_root_path` |

只有 local candidate 非空且實際存在時，才覆寫 runtime attribute。

### 12.3 Explicit override

使用者傳入的 `image_path`、`weight_path`、`pp_num_gpus`、`device` 最後覆寫 saved/localized config。

`pp_num_gpus` 必須大於 0。

### 12.4 ROI sampling config

ROI limit 優先序：

```text
explicit max_rois_per_dxitem
-> saved max_rois_per_dxitem
-> saved legacy max_rois_per_case
```

sampling mode 優先序：

```text
explicit roi_sampling_mode
-> saved roi_sampling_mode
-> "all"
```

mode 必須在支援清單內。

seed 優先序：

```text
explicit valid_sampling_seed
-> saved valid_sampling_seed
-> 42
```

已確認：mode 為 `all` 時，即使 ROI limit 非空，仍保留全部 ROI assignments。

### 12.5 Generation config

`max_new_tokens` 與 `eval_prompt_batch_size` 的 explicit override 都必須大於 0。

缺少 saved 值時分別預設：

```text
max_new_tokens = 256
eval_prompt_batch_size = 6
```

AMP 優先 explicit；saved config 完全沒有欄位時預設 `True`。

## 13. `preflight` 完整流程

### 13.1 檔案與輸出檢查

- input metadata 必須存在。
- input/output absolute path 不得相同。
- output 已存在且 `overwrite=False` 時失敗。
- output parent directory 必須已存在。

### 13.2 Metadata 與 collision

1. 載入 metadata object。
2. 驗證 case identities。
3. 驗證 existing `generated_report` collision。

### 13.3 Config 與 DxItem universe

1. 載入並本地化 config。
2. metadata 必須有 top-level list `DxItem_list`。
3. 若 config 已有 truthy `DxItem_list`，必須與 metadata list 完全相等，包括順序。
4. config 無清單時，複製 metadata list。

### 13.4 Runtime paths

永遠要求 `weight_path` 存在。

只有 `input_img=True` 時要求 `image_path` 存在。

缺少任一必要 path 時一次回報 missing mapping。

### 13.5 Targetless dataset

preflight 直接建立 `multiROI2DxResultDataset`：

```text
split = "valid"
require_targets = False
metadata_path = input metadata
```

這表示：

- 不要求輸入含 `structured_report` targets。
- 使用 validation transform，沒有 train augmentation。
- `random_k` 使用 deterministic per-case、per-DxItem seed。
- dataset 的 DxItem/ROI 結構驗證仍完整執行。
- 每個 case 仍必須至少有一個 active DxItem。

### 13.6 Preflight 統計

回傳：

- case count。
- DxItem list。
- active case-DxItem pair count。
- raw ROI assignment count。
- effective sampled ROI assignment count。
- config/checkpoint/trainer/adapter paths。
- checkpoint epoch。
- sampling 與 generation 設定。
- PP GPU 數與 device。
- `structured_report_required=False`。

raw/effective ROI 統計都是 assignment counts，不是唯一實體 ROI counts。

## 14. Model 建立

`_build_model` 前置條件是已成功 `preflight()`。

### 14.1 Device 檢查

- config device type 為 CUDA 時，系統必須有 CUDA。
- 若指定 `pp_num_gpus`，不得大於可見 CUDA device count。

### 14.2 組裝

```text
DownstreamRepGenVLM.from_config(
    cfg,
    load_criterion=False
)
-> load_lora_weights(adapter_path)
-> model.eval()
```

inference 不建立 criterion，也不讀取 trainer optimizer state。

base VLM 仍從 localized `weight_path` 載入，LoRA topology 必須與 checkpoint adapter 匹配。

## 15. Prediction 主流程

### 15.1 前置條件

`cfg` 與 targetless dataset 必須已由 preflight 建立。

### 15.2 Inference context

整體 loop 位於：

```text
torch.inference_mode()
```

只有 config `amp=True` 且 device type 是 CUDA 時，才再使用 BF16 autocast。

### 15.3 逐 case 執行

pipeline 直接依 dataset index 循序取 case，不建立 DataLoader：

```text
for dataset_idx in range(len(dataset)):
    case = dataset[dataset_idx]
```

每個 case：

1. `case.global_idx` 不得與先前 prediction 重複。
2. 建立一次 `case_context = model.prepare_case_loss_context(case)`。
3. 呼叫 `generate_outputs(batch_cases=[case], ...)`。
4. 從輸出取回完全相同 `case.global_idx` 的 case result。
5. case result 必須是 dictionary。
6. 保存於 predictions mapping。
7. 刪除大物件 references，讓後續 case 可回收記憶體。

模型內部負責同 signature DxItems 的 cached batching 與 fallback。

### 15.4 Prediction mapping

```text
predictions = {
    sample_idx: {
        DxItem: {
            "pred_txt": string,
            "gt_txt": ""
        }
    }
}
```

targetless dataset 下 `gt_txt` 通常是空字串；回寫只使用 `pred_txt`。

## 16. `run` 完整流程

```text
preflight
-> require output_metadata_path
-> set_seed(valid_sampling_seed)
-> build model and predict all cases
-> validate and apply generated reports in memory
-> assert generated_count == preflight active pair count
-> atomically write output metadata
-> return combined summary
```

`set_seed` 在 preflight 後、實際模型建立與 generation 前呼叫。validation `random_k` 本身有 pair-specific RNG；全域 seed 同時穩定其他 Python、NumPy 與 Torch 隨機來源。

若最後 generated count 與 preflight expected count 不同，output file 不得寫出。

## 17. CLI 契約

### 17.1 必要參數

```text
--checkpoint-dir
--input-metadata
```

除非使用 `--preflight-only`，否則也必須提供：

```text
--output-metadata
```

### 17.2 Boolean options

- `--overwrite`：允許覆寫 output file 與 case 內既有 generated report。
- `--preflight-only`：只執行檢查及印出摘要，不建模型、不生成、不寫 output。
- `--amp`／`--no-amp`：使用 `BooleanOptionalAction`，未指定時保留 saved/default 行為。

### 17.3 CLI 回傳

`main` 將 summary 以 UTF-8 friendly、indent 2 JSON 印至 stdout，成功回傳 process exit code 0。

## 18. 安全與失敗原則

以下問題必須在生成前或寫檔前阻止流程：

- checkpoint/config/input 不存在。
- snapshot hash 不符。
- output path collision。
- output directory 不存在。
- input/output 指向同一檔案。
- duplicate/unhashable `sample_idx`。
- config/metadata DxItem list 不一致。
- base weight/image root 不存在。
- 不支援的 sampling mode。
- GPU request 超過可見數量。
- prediction missing、unexpected、empty 或 case identity 不一致。
- generated count 與 expected count 不一致。

pipeline 不應以 partial output 或 placeholder text 繼續。

## 19. 必須維持的不變量

1. inference active DxItems 必須由 ROI `DxPair` 定義。
2. `structured_report` 在 inference 可完全不存在。
3. 每個 case 仍必須至少有一個 active DxItem。
4. sampling 必須使用 validation 語義；`random_k` 必須 deterministic。
5. `all` sampling mode 必須忽略 per-DxItem ROI limit。
6. prediction 必須和 expected `(sample_idx, DxItem)` 集合一一對應。
7. 空 prediction 不得寫入 output。
8. 未指定 overwrite 時不得覆蓋現有 output file 或非空 generated report。
9. input metadata file 本身不得原地修改。
10. final metadata 必須以 atomic replace 寫入。
11. 新版 snapshot 必須在載入前驗證 SHA-256。
12. inference 必須只載入 LoRA adapter，不建立 criterion 或恢復 optimizer。
13. generation 必須維持 greedy deterministic semantics。
14. generated report count 必須與 preflight active pair count 相同。

## 20. 現行邊界與非保證項目

- `trainer_checkpoint_path` 被驗證及回報，但 inference 不讀取其中的 trainer state。
- `checkpoint_dir` 必須存在，即使 explicit config/checkpoint path 指向其他位置。
- output parent directory不會自動建立。
- `overwrite=True` 同時允許覆寫 output file 與 metadata 內既有 generated report，沒有分成兩個獨立權限。
- prediction loop 是逐 case，不使用 DataLoader batching；同 case 內相同 ROI signature 的 DxItems 可由模型 batching。
- pipeline 不執行 evaluator，也不保存生成品質 metrics。
- `_case_active_dxitems` 對 malformed nested blocks 採較寬鬆的跳過，但隨後 dataset preflight 會執行較嚴格的 metadata validation。
- input metadata object 會在記憶體中被修改；若後續 atomic write 失敗，磁碟上的 input/output 仍不會被 partial 更新。

這些是現行行為邊界。若未來要支援 streaming output、case batching、sampling generation 或 in-place metadata 更新，必須先修訂本準則及相應安全條件。


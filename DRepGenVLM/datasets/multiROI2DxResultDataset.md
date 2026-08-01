# `multiROI2DxResultDataset.py` 實作準則

## 1. 文件目的與適用範圍

本文件描述 `multiROI2DxResultDataset.py` 的現行行為、資料契約、抽樣規則與失敗條件，並作為後續修改此模組時應維持的實作準則。

本模組的主要責任是：

- 讀取 case-level metadata。
- 驗證 `DxItem_list`、`structured_report` 與 ROI `DxPair` 的關係。
- 依每個 `DxItem` 獨立收集及抽樣 ROI。
- 對相同實體 ROI 去除重複影像 I/O。
- 建立模型與 evaluator 共用的 `Case`、`ROI` 資料物件。
- 提供以 ROI-to-DxItem assignment 為單位的資源計數。

本模組不負責：

- 將不同 case padding 成張量 batch。
- 建立 VLM prompt 或 tokenizer 輸入。
- 計算 loss、生成文字或評估預測。
- 修補不合法 metadata；不合法的關係必須明確失敗或留下警告。

## 2. 核心名詞與符號

以下符號只使用純文字表示，以維持標準 Markdown 相容性。

```text
C                    所有 case 的集合
D                    metadata.DxItem_list 宣告的 DxItem 有序集合
R_c                  case c 中依 metadata 順序出現的所有實體 ROI
keys(DxPair_r)       ROI r 宣告的 DxItem key 集合
R_(c,d)              case c 中分派給 DxItem d 的有序 ROI record 序列
K                    max_rois_per_dxitem
S_mode(R, K)         指定抽樣模式對 R 的結果
R~_(c,d)             抽樣後的 DxItem ROI record 序列
U_c                  所有 R~_(c,d) 依 global_idx 合併後的實體 ROI 聯集
```

active DxItem 的定義如下：

```text
d 是 case c 的 active DxItem
<=>
至少存在一個 ROI r，使 d 位於 keys(DxPair_r) 中。
```

只有 active DxItem 會成為模型 forward 單位。`structured_report` 中有 target、但沒有任何 ROI `DxPair` 指向它的 DxItem，不會產生獨立 forward。

## 3. 公開資料型別

### 3.1 `ROI`

`ROI` 是完成抽樣及必要影像載入後的模型輸入物件。

| 欄位 | 型別 | 語義 |
| --- | --- | --- |
| `global_idx` | `int` | metadata 中 ROI 的全域識別值，也用於實體 ROI 去重 |
| `image` | `PIL.Image.Image` 或 `None` | 經 RGB 轉換與 split-specific transform 後的影像 |
| `mpp` | `float` 或 `None` | 每 pixel 的實體尺度；若來源為 list/tuple，只採第一個元素 |
| `cxcywh` | 四元素 tuple 或 `None` | ROI 中心與寬高位置資訊 |
| `roi_wh` | 二元素 tuple | ROI 原始寬高；缺失時為 `(0.0, 0.0)` |

`ROI.image` 是否為 `None` 由 `input_img` 與 `roi_path` 共同決定：

```text
image 會被載入 <=> input_img 為 True 且 roi_path 不是 None
```

### 3.2 `ROIRecord`

`ROIRecord` 是 frozen dataclass，用於影像 I/O 前的輕量 metadata 分組及抽樣。

它與 `ROI` 的主要差異是保存 `roi_path`，但尚未開啟影像。這項設計確保未被抽中的 ROI 不會產生影像 I/O 與 augmentation 成本。

### 3.3 `Case`

| 欄位 | 語義 |
| --- | --- |
| `global_idx` | 來源 case 的 `sample_idx`，同時是模型、evaluator 與 inference prediction 的主索引 |
| `case_id` | 轉為字串的 case 識別值 |
| `rois` | 抽樣後實際使用之 ROI 的唯一聯集，只供診斷與資源統計 |
| `DxItem_rois` | active `DxItem -> ordered List[ROI]`；模型輸入必須以此欄位為準 |
| `DxItem_targets` | `DxItem -> DxResultTxt`；供 SFT 與文字指標使用 |
| `DxItem_target_classes` | `DxItem -> DxResultCls`；供權威臨床指標使用 |

重要不變量：

- `Case.rois` 不是 case 內所有原始 ROI，而是抽樣後各 DxItem ROI 的實體聯集。
- `Case.rois` 不得取代 `Case.DxItem_rois` 作為模型輸入。
- 同一個 `ROI` 物件可同時出現在多個 `DxItem_rois` list 中。
- `DxItem_targets` 可包含 inactive DxItem 的 target；這不代表該 DxItem 應執行 forward。

## 4. 建構參數與相容性

### 4.1 主要參數

| 參數 | 現行預設值 | 行為 |
| --- | --- | --- |
| `image_path` | 無 | `roi_path` 的根目錄 |
| `metadata_path` | 無 | case metadata JSON |
| `split` | 無 | `train` 啟用 augmentation；其他值使用空 transform |
| `input_img` | `True` | 是否載入影像 |
| `input_loc` | `True` | 是否讀取 `mpp` 與 `cxcywh` |
| `level_key` | `main_info` | 每個 ROI 中選用的解析度／層級 metadata key |
| `max_rois_per_dxitem` | `None` | 每個 case、每個 DxItem 的 ROI 上限 |
| `roi_sampling_mode` | `all` | `all`、`random_k`、`tail_k` 或 `head_k` |
| `valid_sampling_seed` | `42` | validation `random_k` 的基底 seed |
| `require_targets` | `True` | 是否要求完整 `structured_report` targets |
| `max_rois_per_case` | `None` | 舊版相容別名，實際語義已是 per-DxItem 上限 |

### 4.2 ROI 上限的舊版相容規則

若新舊參數都提供且數值不同，建構必須失敗：

```text
max_rois_per_dxitem != max_rois_per_case
=> ValueError
```

若只有舊參數存在，將其值視為 `max_rois_per_dxitem`。後續程式不得把它重新解釋成全 case 的唯一 ROI 上限。

## 5. Metadata 契約

### 5.1 最小結構

```json
{
  "DxItem_list": ["Histologic_Type", "Histologic_Grade"],
  "case_list": [
    {
      "sample_idx": 0,
      "case_id": "case-001",
      "structured_report": {
        "DxItems": {
          "Histologic_Type": {
            "DxResultTxt": "...",
            "DxResultCls": "..."
          }
        }
      },
      "tissue_blocks": [
        {
          "stains": [
            {
              "roi_list": [
                {
                  "global_idx": 100,
                  "DxPair": {
                    "Histologic_Type": "..."
                  },
                  "main_info": {
                    "roi_path": "relative/path.png",
                    "mpp": 0.25,
                    "cxcywh": [0.5, 0.5, 0.2, 0.2],
                    "roi_wh": [1024, 1024]
                  }
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

`DxPair` 的 value 不參與本模組的分組；目前只使用其 key 判斷 ROI 應分派給哪些 DxItems。

### 5.2 `DxItem_list` 驗證

`DxItem_list` 必須：

- 是非空 list。
- 每個元素都是非空字串。
- 不得包含重複項目。

### 5.3 Case 與 target 驗證

每個 `case_list` 元素必須是 object。

當 `require_targets=True` 時：

- `structured_report` 必須是 object。
- `structured_report.DxItems` 必須是 object。
- 每個 DxItem target 必須是 object。
- 每個 target 必須同時含 `DxResultTxt` 與 `DxResultCls`。
- ROI `DxPair` 指向的 DxItem 必須存在對應 target。

當 `require_targets=False` 時：

- 缺少 `structured_report` 或 `DxItems` 時可使用空 target mapping。
- 若 target 存在，仍不得含未在頂層 `DxItem_list` 宣告的 DxItem。
- ROI 與 active DxItem 的結構驗證仍然執行。

不論 `require_targets` 為何，每個 case 都必須至少有一個合法 active DxItem。

### 5.4 ROI 關係驗證

- `tissue_blocks` 必須是 list。
- 每個 tissue block 必須包含 `stains` list。
- 若 stain 有 `roi_list`，該欄位必須是 list。
- 每個 ROI 必須是 object。
- `DxPair=None` 或缺少 `DxPair` 的 ROI 不會成為 active assignment。
- 非 `None` 的 `DxPair` 必須是 object。
- `DxPair` 的每個 key 必須已在頂層 `DxItem_list` 宣告。

### 5.5 可疑 reference 警告

初始化後會額外尋找明顯可疑的文字 reference，但不會移除 case：

- `DxResultTxt` 不是字串或只有空白。
- `Histologic_Type` 移除非英文字母後不足三個字母。

這些紀錄保存在 `invalid_reference_records`，並以 `[MetadataValidation][WARN]` 輸出。它們不是自動修復機制。

## 6. ROI assignment 計數

### 6.1 Raw count

`get_raw_roi_count(idx)` 計算：

```text
raw_count(c) = sum over ROI r in case c of len(keys(DxPair_r))
```

因此，同一實體 ROI 若分派給三個 DxItems，raw count 增加 3，不是 1。

### 6.2 Effective count

先對每個 DxItem 計算：

```text
n_(c,d) = number of ROI records assigned to d
```

再依 sampling mode 計算：

```text
if K is None or K <= 0 or mode == "all":
    effective_count(c) = sum_d n_(c,d)

if mode in {"random_k", "tail_k", "head_k"}:
    effective_count(c) = sum_d min(n_(c,d), K)
```

`effective_count` 仍是 assignment 數，不是 `Case.rois` 的唯一實體 ROI 數。

已確認的準則：`roi_sampling_mode="all"` 必須忽略 `max_rois_per_dxitem`，保留全部 assignments。

## 7. 抽樣演算法

### 7.1 每個 DxItem 獨立抽樣

對每個 case 及每個宣告 DxItem，先依 metadata traversal 順序建立：

```text
R_(c,d) = [ROIRecord r | d in keys(DxPair_r)]
```

每個 `R_(c,d)` 獨立套用抽樣，不能先對 case 全域抽一組 ROI 再分配。

### 7.2 各 mode 的精確語義

```text
K is None or K <= 0:
    return entire input list

len(R) <= K:
    return entire input list

mode == "random_k":
    sample K distinct indices
    sort sampled indices
    return elements in original relative order

mode == "tail_k":
    return final K elements

mode == "head_k":
    return first K elements

mode == "all":
    return entire input list, even when len(R) > K
```

不支援的 mode 只會在需要查詢 effective count 或實際超過上限而進入 `_sample_rois` 時拋出 `ValueError`。

### 7.3 Train 與 validation 的亂數差異

`train + random_k` 使用模組層級的 Python `random` 狀態，因此可隨 epoch／呼叫順序變動，並由外部全域 seed 與 checkpoint RNG restore 管理。

`valid + random_k` 對每個 `(case index, DxItem index)` 建立獨立 RNG：

```text
pair_seed = valid_sampling_seed
            + case_index * max(1, number_of_declared_DxItems)
            + DxItem_index
```

因此同一 metadata 順序、同一 seed 與同一 DxItem 順序下，validation subset 應跨 epoch 穩定。

## 8. `__getitem__` 完整流程

### 步驟 1：遍歷 metadata 並建立輕量 records

對每個 ROI：

1. 讀取 `roi_sample[level_key]`；缺少時使用空 object。
2. 取得 `roi_path`。
3. 僅在 `input_loc=True` 時解析 `mpp` 與 `cxcywh`。
4. `mpp` 若是 list/tuple，僅採第一個元素並轉成 `float`。
5. `roi_wh` 與 `input_loc` 無關；缺少時使用 `(0.0, 0.0)`。
6. 建立一個 `ROIRecord`。
7. 對 `DxPair` 的每個 key，將同一 record 追加到該 DxItem 的 list。

### 步驟 2：逐 DxItem 抽樣

只保留至少有一個 record 的 DxItem。抽樣後得到：

```text
sampled_records_by_dxitem: DxItem -> ordered List[ROIRecord]
```

此 mapping 同時界定 active model forward units。

### 步驟 3：建立實體 ROI 聯集

所有抽樣後 records 依 `global_idx` 去重：

```text
U_c = ordered unique union of all R~_(c,d)
```

若相同 `global_idx` 對應到不完全相同的 `ROIRecord`，代表同一識別值具有互相矛盾的 level metadata，必須拋出 metadata error。

去重順序遵循 `DxItem_list` 與各抽樣 list 的首次出現順序。

### 步驟 4：載入與轉換影像

對 `U_c` 中每個 record 最多開啟影像一次：

```text
open image_path / roi_path
-> convert("RGB")
-> apply split transform
-> leave with-context and retain transformed PIL image
```

相同 ROI 被多個 DxItems 使用時，它們共享同一份 augmentation 結果，不會各自獨立隨機翻轉或旋轉。

### 步驟 5：重建 `DxItem_rois`

以 `global_idx -> ROI` mapping 把每個抽樣 record 序列重建為模型用的 `List[ROI]`。順序不得改變。

### 步驟 6：收集 targets

依頂層 `DxItem_list` 順序，從存在的 report targets 分別收集：

```text
DxItem_targets[DxItem] = DxResultTxt
DxItem_target_classes[DxItem] = DxResultCls
```

`DxResultCls` 是臨床 evaluator 的權威 reference；不得由 `DxResultTxt` 重新推論替代。

### 步驟 7：建立 `Case`

```text
Case.global_idx = sample["sample_idx"]
Case.case_id = str(sample["case_id"])
```

`sample_idx` 不在 dataset 內被強制轉成 `int`；下游需將它視為可作 dictionary key 的 identity。

## 9. 影像 augmentation

`split == "train"` 時依序套用：

1. 50% horizontal flip。
2. 50% vertical flip。
3. 從 `0`、`90`、`180`、`270` 度均勻隨機選一個角度旋轉，且 `expand=True`。

其他 split 使用空 `transforms.Compose`，不做 augmentation。

`RandomDiscreteRotation` 使用 Python `random.choice`，不使用 Torch RNG。

## 10. DataLoader 介面

`collate_cases(batch)` 原樣回傳 `list(batch)`，不做 padding、stacking 或 device transfer。

所以 DataLoader batch 的語義是：

```text
List[Case]
```

而不是單一 tensor dictionary。

## 11. `from_config` 行為

- 只接受 `split="train"` 或 `split="valid"`。
- 依 split 選擇 `train_metadata_path` 或 `valid_metadata_path`。
- 新舊 ROI limit 同時存在且衝突時失敗。
- 將 dataset 的 `DxItem_list` 回寫到 `cfg.DxItem_list`，供 model、trainer 與 evaluator 使用。
- `from_config` 未顯式傳入 `require_targets`，因此訓練／驗證 dataset 預設要求完整 targets。
- targetless inference 由 inference pipeline 直接建構 dataset 並指定 `require_targets=False`。

## 12. 必須維持的不變量

後續修改原則上不得違反下列規則：

1. ROI 必須先依 `DxPair` 分組，再對每個 DxItem 獨立抽樣。
2. active DxItem 必須由 ROI `DxPair` 定義，不得由 target 是否存在定義。
3. `all` mode 必須忽略 ROI 上限。
4. validation `random_k` 必須維持 per-case、per-DxItem deterministic。
5. 多 DxItem 共用的實體 ROI 必須只載入與 transform 一次。
6. `DxItem_rois` 的有序 ROI signature 必須穩定，因為模型依此判斷視覺 prefix 能否共享。
7. `Case.rois` 只能用於資源與診斷，不能成為所有 DxItems 的共同模型輸入。
8. raw/effective ROI count 必須維持 assignment-count 語義。
9. `DxResultTxt` 與 `DxResultCls` 必須分開保存。
10. 非法 DxItem 關係不得被靜默忽略或自動修補。

## 13. 已知邊界與非保證項目

- 本模組未驗證 `roi_path` 指向的檔案一定存在；實際選中並開圖時才會由 `Image.open` 失敗。
- `mpp` list/tuple 只使用第一個值，沒有檢查 x/y MPP 是否一致。
- `cxcywh` 與 `roi_wh` 沒有在本模組驗證長度、範圍或單位。
- dataset 不執行 tokenizer 長度限制或 ROI 數造成的模型 context 長度檢查。
- `global_idx` 重複但 metadata 完全相同時會視為同一實體 ROI；不同時才會失敗。
- `split` 建構子本身只特判字串 `train`；只有 `from_config` 嚴格限制 train/valid。

這些是現行行為的邊界描述，不代表應由本模組自行補救。


# `DRGVLMLoss.py` 實作準則

## 1. 文件目的

本文件描述 `DRGVLMLoss.py` 的現行輸入契約、causal language modeling 位移、cross-entropy 計算與數值防護，並作為後續修改此 loss 模組的實作準則。

`DRGVLMLoss` 只負責 token-level supervised fine-tuning loss。它不負責：

- 建立 prompt。
- 決定哪些 token 應被遮罩。
- 將 case 或 DxItem 加權。
- 生成文字或計算臨床指標。
- 執行 backward、梯度累積或 gradient clipping。

上述責任分別位於 model 與 trainer。

## 2. 輸入與輸出契約

### 2.1 `logits`

```text
shape = (B, S, V)
```

其中：

```text
B = batch size
S = sequence length
V = vocabulary size
```

內容必須是尚未正規化的 raw logits，不應先做 softmax 或 log-softmax。

### 2.2 `labels`

```text
shape = (B, S)
dtype 通常為 torch.long
```

每個位置的語義：

```text
labels[b, t] == -100
    該位置不參與 cross-entropy

labels[b, t] in [0, V)
    該位置是要預測的 tokenizer vocabulary ID
```

目前由 `DownstreamRepGenVLM` 保證 user prompt、影像／位置 prefix、chat template 前綴及 padding 都標為 `-100`。

### 2.3 `context`

`context` 是只供錯誤診斷使用的字串，例如：

```text
case_id=<id>, DxItem=<name>
```

它不參與數值計算。

### 2.4 回傳值

```python
{
    "total_loss": ce_loss
}
```

`total_loss` 是 scalar tensor，保留 autograd graph。

## 3. 輸入形狀驗證

forward 一開始必須檢查：

```text
logits.ndim == 3
labels.ndim == 2
logits.shape[:2] == labels.shape
```

任一條件不成立時拋出 `ValueError`。loss 不得嘗試 broadcast、截斷或 padding 來掩蓋 shape mismatch。

## 4. Causal language modeling 位移

模型在位置 `t` 的 logits 用來預測位置 `t + 1` 的 token，因此計算前必須位移：

```text
shift_logits = logits[:, 0:S-1, :]
shift_labels = labels[:, 1:S]
```

位移後形狀：

```text
shift_logits.shape = (B, S-1, V)
shift_labels.shape = (B, S-1)
```

這代表原序列最後一個 logit 沒有下一個 label，而原序列第一個 label 沒有上一個 logit，兩者都不進入 loss。

## 5. Cross-entropy 定義

令有效監督集合為：

```text
Omega = all (b, t) where shift_labels[b, t] != -100
```

無 label smoothing 時，loss 可寫成：

```text
loss = mean over (b, t) in Omega of:
       -log softmax(shift_logits[b, t])[shift_labels[b, t]]
```

實作等價於：

```python
F.cross_entropy(
    shift_logits.reshape(-1, V).to(torch.float32),
    shift_labels.reshape(-1),
    ignore_index=-100,
    label_smoothing=self.label_smoothing,
)
```

重要語義：

- reduction 使用 PyTorch `cross_entropy` 的預設 `mean`。
- 分母是非 `-100` target 的數量。
- 不依 case、DxItem 或 ROI 數額外加權。
- logits 無論原始 dtype 為何，都在 CE 前轉為 float32。
- labels 不改變 dtype 或 device。

## 6. Label smoothing

`label_smoothing` 由建構子轉成 `float`，預設為 `0.0`。

當其為 `epsilon` 時，使用 PyTorch `F.cross_entropy` 的標準 label-smoothing 定義；本模組不自行重實作 target distribution。

後續修改不得在本 loss 內加入 per-DxItem smoothing、class weight 或 ROI weight，除非整體演算法準則同步更新。

## 7. 數值有限性檢查

### 7.1 `_is_finite`

檢查 tensor 中所有元素是否都是 finite：

```text
finite <=> no NaN and no positive/negative infinity
```

結果 detach 並移到 CPU 後轉成 Python bool。

### 7.2 `_finite_summary`

若發現非有限值，診斷摘要包含：

- shape。
- dtype。
- 非有限元素數量。
- 若仍有有限元素，列出 float32 下的 finite min/max。
- 若全為非有限值，標示 `all_nonfinite=True`。

### 7.3 Fail-fast

CE 計算後必須立即驗證：

```text
isfinite(ce_loss) must be True
```

否則拋出 `RuntimeError`，錯誤訊息包含 tensor 名稱、caller context 與 finite summary。

常見觸發原因包括：

- labels 全部是 `-100`，沒有有效 target。
- upstream logits 已有 NaN/Inf。
- 模型或輸入數值不穩定。

此模組不得將非有限 loss 替換成 0，也不得跳過該 pair。

## 8. `from_config`

`from_config` 只讀取：

```text
cfg.label_smoothing
```

若 config 沒有該欄位，使用 `0.0`。

目前沒有其他 loss hyperparameters。

## 9. 與 model、trainer 的責任邊界

### 9.1 Model 必須保證

- `logits` 與 `labels` 的 batch/sequence dimensions 一致。
- 只有 assistant answer 應有非 `-100` labels。
- cached 與 uncached path 應產生語義等價的 labels。
- 每個 `(case, DxItem)` 至少有一個有效監督 token。

### 9.2 Loss 必須保證

- 標準 causal shift。
- `ignore_index=-100`。
- float32 cross-entropy。
- optional label smoothing。
- scalar loss finite check。

### 9.3 Trainer 必須保證

- pair loss 的 case/DxItem/accumulation 加權。
- backward。
- gradient finite check 與 clipping。
- optimizer 與 scheduler step。

## 10. 必須維持的不變量

1. loss 必須是 causal next-token prediction，不能改成同位置 token prediction。
2. user、image、位置、template 與 padding token 必須可透過 `-100` 完全排除。
3. CE 必須使用 raw logits。
4. CE 必須在 float32 計算，以降低低精度 softmax/CE 的數值風險。
5. reduction 必須維持對有效 token 的平均。
6. 本模組不得自行進行 case 或 DxItem weighting。
7. 非有限 loss 必須立即拋錯，不能靜默修正。
8. 回傳 key 必須保留 `total_loss`，以符合 model、trainer 與 metrics tracker 契約。

## 11. 現行邊界

- 本模組沒有顯式檢查 `S >= 2`；過短序列會由後續 CE 行為與 finite check 暴露。
- 本模組沒有預先檢查是否存在非 `-100` label。
- 本模組沒有檢查有效 label 是否落在 `[0, V)`；越界由 PyTorch CE 拋錯。
- `_debug_print`、`log_print`、`Any` 與 `Optional` 目前被 import，但不參與主計算。

這些是現行行為描述，不應被解讀為允許吞掉 upstream 資料錯誤。


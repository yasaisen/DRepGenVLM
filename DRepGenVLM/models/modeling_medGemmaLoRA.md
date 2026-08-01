# `modeling_medGemmaLoRA.py` 實作準則

## 1. 文件目的與模組角色

本文件描述 `modeling_medGemmaLoRA.py` 中 `DownstreamRepGenVLM` 的現行模型組裝、prompt 建立、訓練 forward、共享視覺快取、文字生成與 LoRA 存取行為，並作為後續修改的實作準則。

此模型的基本計算單位不是單一 case，而是：

```text
(case, active DxItem)
```

每個 active DxItem 只能使用 dataset 在 `Case.DxItem_rois[DxItem]` 中指派給它的有序 ROI 序列。

本模組的主要責任為：

- 載入並掛載 frozen VLM 與 processor。
- 注入、驗證及存取 LoRA adapter。
- 將一個 `(case, DxItem)` 建成多模態 chat prompt。
- 建立只監督 assistant answer 的 SFT labels。
- 選擇完整 VLM forward 或 detached shared-vision-prefix forward。
- 對單一 pair 計算 loss，或對 batch 彙總 loss。
- 以 greedy decoding 產生各 active DxItem 的文字答案。

## 2. 核心符號與張量形狀

```text
c                   一個 Case
d                   c 的一個 active DxItem
R_(c,d)             Case.DxItem_rois[d] 的有序 ROI 序列
sig(c,d)            R_(c,d) 中 global_idx 組成的有序 tuple
S_ids               processor input_ids 的 token 長度
S_merged            影像 token 展開後、第一層文字 decoder 前的序列長度
H                   text hidden size
V                   tokenizer vocabulary size
q_start_ids         最後一個 SEP token 之後的 token-space index
P                   merged-space 視覺 prefix 長度
```

主要張量：

| 張量 | 形狀 | 語義 |
| --- | --- | --- |
| `input_ids` | `(1, S_ids)` | processor 產生的 token IDs |
| `merged_embeds` | `(1, S_merged, H)` | 影像展開並投影後，第一層文字 decoder 的輸入 |
| `inputs_embeds` | `(B, S, H)` | cached training/generation 的重建 embedding 序列 |
| `attention_mask` | `(B, S)` | 完整重建後的 attention mask |
| `token_type_ids` | `(B, S)` 或不存在 | 部分 Gemma 多模態模型的 token type |
| `logits` | `(B, S, V)` | causal language model logits |
| `labels` | `(B, S)` | 非監督位置為 `-100` |

## 3. 初始化狀態

`DownstreamRepGenVLM.__init__` 設定：

| 屬性 | 初始值 | 語義 |
| --- | --- | --- |
| `device` | 指定裝置；預設可用 CUDA 否則 CPU | processor 輸出移入的 home device |
| `use_text_gradient_checkpointing` | `False` | 目前保留狀態，主流程未啟用 |
| `require_cached_input_grad` | `False` | 目前 cached inputs 不主動設 `requires_grad` |
| `use_shared_vision_cache` | `True` | 之後由 config 覆寫 |
| `use_vision_lora` | `False` | 之後由 `apply_lora` 設定 |
| `eval_prompt_batch_size` | `6` | cached generation 中每批 DxItem prompt 數 |

`SYSTEM_PROMPT` 目前為空字串，且現行 message builder 未使用 system turn。

## 4. Training mode 控制

### 4.1 `train(mode)` 的特別語義

呼叫 `DownstreamRepGenVLM.train(True)` 時：

1. 先依一般 PyTorch 規則切換整個 wrapper。
2. 若已掛載 `vlm_model`，立即將完整 VLM 設回 eval mode。
3. 若 `mode=True`，遍歷 VLM 內所有模組，只將 PEFT `lora_dropout` 設為 training mode。

因此：

- frozen VLM backbone 的原生 dropout 等 stochastic module 保持 eval。
- LoRA parameter 是否可訓練由 `requires_grad` 決定，不依賴 `Module.training`。
- LoRA dropout 是訓練期唯一明確重新開啟的 stochastic module。

## 5. 掛載 VLM 與解析 text backbone

### 5.1 `init_vlm_model`

必要條件：`vlm_processor` 必須有 `.tokenizer`。

初始化內容：

- 保存 processor。
- 讀取 `model_type` 與 model dtype。
- 優先從 `config.text_config.hidden_size` 讀取 hidden size，否則用 `config.hidden_size`。
- 優先從 `config.boi_token_id` 取得 image beginning token；缺少時由 tokenizer 查詢 `<start_of_image>`。
- 以 `object.__setattr__` 掛載原始 VLM，使套用 LoRA 前它不進入 wrapper 的 module tree。

`max_senLen` 參數目前沒有保存，也沒有在此模型中執行 prompt truncation。

### 5.2 `_resolve_text_backbone`

套用 PEFT 後，模型使用 breadth-first search 尋找具有非空 `.layers` 的文字 decoder。

搜尋 child attribute：

```text
language_model
model
text_model
transformer
base_model
```

找不到時必須失敗，因為 shared vision cache 與 token embedding 都依賴解析出的 text backbone。

## 6. LoRA 契約

### 6.1 Adapter 設定

`apply_lora` 使用：

```text
task_type       = CAUSAL_LM
r               = lora_r
lora_alpha      = lora_alpha
lora_dropout    = lora_dropout
target_modules  = lora_target_modules
bias            = none
inference_mode  = False
```

當 `use_vision_lora=False` 時，以 pattern 排除 `vision_tower` 的所有模組。這是必要措施，因為 PEFT 的 target module suffix，例如 `q_proj`，可能同時匹配文字 decoder 與 vision tower。

### 6.2 Scope 驗證

套用後必須檢查：

- trainable LoRA parameter 總數不得為 0。
- `use_vision_lora=False` 時，vision tower 中 trainable LoRA parameter 數必須為 0。
- `use_vision_lora=True` 時，vision tower 中 trainable LoRA parameter 數不得為 0。

任何 scope 不一致都必須失敗，不能只記錄警告。

### 6.3 Module registration

PEFT wrapper 建立後，以一般 `self.vlm_model = peft_model` 設定，使其進入 wrapper 的 module tree。此時：

- frozen base parameters 仍為 `requires_grad=False`。
- optimizer 可透過 `DownstreamRepGenVLM.parameters()` 找到 LoRA parameters。
- `text_backbone` 則以 `object.__setattr__` 保存，避免額外重複註冊同一 module subtree。

### 6.4 已確認的快取互斥規則

```text
use_shared_vision_cache == True
and
use_vision_lora == True
=> ValueError
```

理由是視覺 prefix 在 `torch.no_grad()` 下建立；若允許 vision LoRA，視覺 adapter 不會收到梯度。

## 7. 特殊 token

`init_sep` 向 tokenizer 的 `additional_special_tokens` 依序加入：

| token | 預設字串 | 用途 |
| --- | --- | --- |
| SEP | `<unused0>` | 結束每個 ROI 的文字區塊，並用於定位問題起點 |
| BOC | `<unused1>` | ROI 沒有影像輸入時的替代標記 |

兩個 token 都使用：

```text
AddedToken(normalized=False, special=True)
```

轉換後的 token ID 不得等於 tokenizer 的 unknown token ID。

模型目前沒有呼叫 `resize_token_embeddings`；此設計依賴所用 tokenizer／checkpoint 已能容納這些 special tokens，或 `add_special_tokens` 沒有實際擴張 vocabulary。

## 8. Active DxItem 與 ROI 存取

### 8.1 `_active_dxitems`

active DxItems 依 `case.DxItem_rois` 的 insertion order 回傳，且只包含 ROI list 非空者。

缺少 `DxItem_rois` 時必須失敗；模型不再接受僅有 case-wide `rois` 的舊資料格式。

### 8.2 `_get_dxitem_rois`

- requested DxItem 不在 mapping 中時拋出 `KeyError`。
- mapping 存在但 list 為空時拋出 `ValueError`。
- 成功時原樣回傳有序 ROI list。

### 8.3 ROI signature 分組

```text
sig(c,d) = tuple(roi.global_idx for roi in Case.DxItem_rois[d])
```

只有 signature 完全相同的 DxItems 才會被分在同一 group。

「完全相同」包含：

- ROI 數量相同。
- 每個 `global_idx` 相同。
- 順序相同。

集合相同但順序不同，不得共享視覺 prefix。

## 9. Prompt 建立

### 9.1 單一 ROI 的 user content

對 `R_(c,d)` 中每個 ROI，按原順序建立：

```text
if ROI.image is not None:
    append {"type": "image"}

text = ""
if ROI.image is None:
    text += BOC
if ROI.mpp is not None:
    text += "MPP: <six-decimal float>, "
if ROI.cxcywh is not None:
    text += "cxcywh: (<four six-decimal floats>), "
text += SEP

append {"type": "text", "text": text}
```

所以每個 ROI 一定會產生一個以 SEP 結尾的 text content；只有實際有影像時才產生 image content。

### 9.2 診斷問題

所有 ROI content 後固定追加：

```text
Based on the histological images above, what is the {DxItem}?
```

### 9.3 Training 與 inference message

Training：

```text
user      = ROI contents + diagnostic question
assistant = Case.DxItem_targets[DxItem]
```

Inference：

```text
user = ROI contents + diagnostic question
```

沒有 system message。

### 9.4 Chat template

processor 以 `tokenize=False` 套用 chat template。

若 `model_type == "gemma4"`，額外指定：

```text
enable_thinking = False
```

Training full prompt 使用 `add_generation_prompt=False`；用來定位答案起點的 prompt-only 與 inference prompt 使用 `add_generation_prompt=True`。

## 10. Processor 輸入與裝置

`_encode_inputs` 呼叫 processor：

```text
processor(
    text=text_prompt,
    images=images if images is non-empty else None,
    return_tensors="pt"
)
```

所有 tensor 移到 `self.device`；浮點 tensor 再轉成模型 dtype。非 tensor 欄位原樣保留。

`self.device` 是 VLM pipeline 的 home device，PP 模型內後續跨 GPU 傳輸由 Transformers／Accelerate device map hooks 管理。

## 11. 非快取 SFT 輸入

`build_train_inputs` 會對同一 `(case, DxItem)` 執行兩次 processor：

1. 完整 user + assistant prompt。
2. user-only + generation prompt。

令：

```text
prompt_len = length(prompt-only input_ids)
```

labels 建立方式：

```text
labels = clone(full input_ids)
labels[:, :prompt_len] = -100
labels[attention_mask == 0] = -100
```

因此監督範圍是完整 prompt 中 `prompt_len` 之後的 assistant answer token 與 template 所包含的結尾 token。

## 12. Shared vision cache 原理

### 12.1 目的

同一 case 的多個 DxItems 若使用完全相同的 ROI 序列，影像 tower 與 multimodal projector 的輸出相同。模型可只計算一次這段 prefix，再為不同問題重建文字 suffix。

### 12.2 擷取 merged embeddings

`get_merged_embeds`：

1. 在 `text_backbone.layers[0]` 註冊 forward pre-hook。
2. 於 `torch.no_grad()` 下執行一次完整 VLM forward。
3. 擷取第一層 decoder 收到的 `hidden_states`。
4. 使用 `detach().clone()` 保存。
5. 無論 forward 是否成功都移除 hook。

擷取點已在影像 token 展開及 multimodal projector 之後，形狀為：

```text
(1, S_merged, H)
```

### 12.3 找出視覺 prefix 邊界

在 reference inference prompt 的 `input_ids` 中找最後一個 SEP：

```text
q_start_ids = last_position(input_ids == SEP_ID) + 1
```

從 `q_start_ids` 到序列尾端都是純文字，不再有 image placeholder 展開，所以：

```text
tail_text_len = S_ids - q_start_ids
P = S_merged - tail_text_len
```

視覺 prefix 為：

```text
vision_prefix_embeds = merged_embeds[:, :P, :]
```

若找不到 SEP 或 `P <= 0`，必須失敗。

### 12.4 Mask 一致性

reference `attention_mask` 若不存在，以全 1 建立。

cached path 要求：

```text
length(reference attention_mask) == S_merged
```

若有 `token_type_ids`，也要求：

```text
length(reference token_type_ids) == S_merged
```

不相等時不嘗試猜測對齊，直接要求停用 shared cache 或更換相容 processor/model。

### 12.5 Context 分享

`prepare_case_loss_context` 對每個相同 ROI signature group：

1. 使用 group 第一個 DxItem 建立一次 prefix context。
2. 將同一個 context dictionary 物件指派給 group 中所有 DxItems。

不同 signature group 絕對不能共用 context。

## 13. Cached training 輸入重建

`build_train_inputs_with_vision_cache` 對每個 DxItem：

1. 再建立完整 training prompt，取得 `full_ids`。
2. 建立 inference prompt，取得 `prompt_len`。
3. 從共同的 `q_start_ids` 切出：

```text
suffix_ids = full_ids[:, q_start_ids:]
```

4. 在 `torch.no_grad()` 下用 frozen `embed_tokens` 將 suffix IDs 轉成 embeddings。
5. 拼接：

```text
combined_embeds = concat(
    vision_prefix_embeds,
    suffix_embeds,
    sequence_dimension
)
```

6. 以相同方式拼接 prefix/suffix attention mask。
7. 若任一 prompt 有 `token_type_ids`，reference 與 full prompt 必須同時具有，並以相同邊界拼接。

答案在 suffix 中的起點：

```text
answer_start_in_suffix = prompt_len - q_start_ids
```

答案在 combined sequence 中的起點：

```text
answer_start_in_combined = P + answer_start_in_suffix
```

labels：

```text
labels[:] = -100
labels[:, answer_start_in_combined:] =
    suffix_ids[:, answer_start_in_suffix:]
labels[attention_mask == 0] = -100
```

若答案起點不小於 combined sequence 長度，現行行為是記錄警告並留下全 `-100` labels；後續 loss 通常會因無有效 target 而形成非有限值，並由 loss/trainer 的 finite check 阻止訓練繼續。

## 14. 單一 DxItem loss

`calculate_dxitem_loss` 的前置條件：

- `DxItem` 必須存在於 `Case.DxItem_targets`。
- shared cache 開啟時，必須提供對應 `case_context`。

### 14.1 Cached path

```text
VLM(
    inputs_embeds=combined_embeds,
    attention_mask=rebuilt_mask,
    token_type_ids=rebuilt_token_types if present,
    use_cache=False
)
```

cached embeddings 是 detached，但文字 LoRA parameters 在 decoder forward 中仍建立自己的 autograd path。

### 14.2 Uncached path

```text
VLM(
    **processor_inputs,
    output_hidden_states=False,
    use_cache=False
)
```

最後都將：

```text
outputs.logits
labels
context="case_id=..., DxItem=..."
```

交給 `DRGVLMLoss`。

## 15. Batch loss 彙總

`calculate_loss` 主要供不串流 backward 的 caller，例如 validation。

流程：

1. 將輸入正規化成 `List[Case]`。
2. 對每個 case 取得 active DxItems。
3. 使用 caller 傳入的 contexts，或自行建立 contexts。
4. 逐 `(case, DxItem)` 計算 scalar total loss。
5. 對所有 pairs 直接算術平均：

```text
total_loss = mean(all pair losses)
```

這裡的彙總與 trainer 訓練期的 case-balanced streamed backward 不同；validation 的 `calculate_loss` 是所有 collected pairs 等權平均。

沒有任何 pair 時必須失敗。

回傳的 `output_case_dict` 在 loss path 中只含：

```text
pred_txt = None
gt_txt   = target text
```

## 16. 生成流程

所有 generation 都使用 deterministic greedy decoding：

```text
do_sample = False
top_p = None
top_k = None
use_cache = True
```

### 16.1 Cached suffix tokenization

`_build_inference_suffix_tokens` 對單一 DxItem 呼叫 processor，但只保留最後 SEP 後的：

- `suffix_ids`
- `suffix_attention_mask`
- optional `suffix_token_type_ids`

processor 產生的 pixel tensors 不進入 VLM，因為視覺 prefix 已存在。

該 prompt 的 `q_start` 必須與 reference context 的 `expected_q_start` 完全相同。

### 16.2 Cached batched generation

同一 ROI signature group 依 `eval_prompt_batch_size` 切成 DxItem chunks。

每一 row 都包含：

```text
[left padding] [shared vision prefix] [DxItem-specific question suffix]
```

left padding 長度：

```text
left_padding = max_sequence_length
               - prefix_length
               - suffix_length
```

padding 放在 prefix 前方，使每一 row 的有效 prompt 連續且都結束於相同 index，符合 decoder-only batched generation 的需求。

使用 `inputs_embeds` 呼叫 `generate` 時，現行 Transformers 行為是回傳新生成 token IDs，不含 prompt IDs，所以直接 decode 整列 `generated[row]`。

### 16.3 Uncached generation

逐 active DxItem 執行完整 processor + VLM generate。此時輸出含 prompt token，因此 decode：

```text
generated[0, prompt_len:]
```

### 16.4 Fallback 順序

對每一相同-signature group：

```text
1. cached generation with configured prompt_batch_size
2. if RuntimeError/ValueError/TypeError:
       clear CUDA cache when available
       retry cached generation with prompt_batch_size = 1
3. if retry also fails:
       mark whole case cache_failed
       regenerate all active DxItems through full uncached VLM
```

只要 case 中任一 group 的 sequential cached generation 也失敗，已生成的 partial cached results 會被完整 uncached case results 取代。

### 16.5 生成輸出契約

```text
{
  case.global_idx: {
    DxItem: {
      "pred_txt": decoded_and_stripped_text,
      "gt_txt": target_text_or_empty_string
    }
  }
}
```

targetless inference 時 `gt_txt` 為空字串。

## 17. LoRA checkpoint I/O

### 17.1 儲存

`save_lora_weights(path)` 必須呼叫 PEFT model 的 `save_pretrained(path)`，只保存 adapter 相關 artifact，不保存 trainer、optimizer 或 scheduler state。

### 17.2 載入

`load_lora_weights(path)` 使用既有 PEFT model 的：

```text
load_adapter(
    path,
    adapter_name="default",
    is_trainable=True
)
```

若有 missing 或 unexpected keys，必須失敗，以避免 text-only 與 vision+text adapter 拓撲混用。

載入後若支援 `set_adapter`，必須啟用 `default` adapter。

## 18. `from_config` 組裝順序

1. 讀取並驗證 `use_shared_vision_cache` 與 `use_vision_lora`。
2. 建立 `modelBuilder(weight_path)`。
3. 以 `project_name="DRGVLM"` 載入 VLM、processor 與 PP device map。
4. 強制 SFT causal attention：`use_bidirectional_attention=False`。
5. 使用 `torch.bfloat16` 載入 base model。
6. `init_vlm_model`。
7. `apply_lora`。
8. `init_sep`。
9. 套用 shared cache 與 eval prompt batch size。
10. 若 `load_criterion=True`，由 config 建立 `DRGVLMLoss`。

PP GPU 數只在 `cfg.training_mode == "PP"` 時傳入 model builder。

## 19. 必須維持的不變量

1. 模型 forward 單位必須是 `(case, active DxItem)`。
2. 每個 DxItem 只能使用自己的 `Case.DxItem_rois`。
3. ROI signature 必須包含順序；只有 signature 完全相同才能共享 prefix。
4. shared vision prefix 必須 detached，且 shared cache 不得與 vision LoRA 同時啟用。
5. frozen VLM backbone 必須維持 eval；訓練期只開啟 LoRA dropout。
6. user、image、位置、template 與 padding token 不得參與 SFT CE。
7. cached path 重建後的 embeddings、attention mask 與 token type 長度必須一致。
8. 找不到 SEP 或 prefix 邊界不合理時不得猜測。
9. generation 必須保持 greedy decoding，除非未來準則明確修改。
10. cached generation 失敗時必須保留 sequential retry 與 full-VLM fallback。
11. LoRA topology mismatch 不得靜默載入。
12. `DxResultCls` 不得由模型文字 prompt 或 loss path重新推導。

## 20. 現行邊界與非保證項目

- `max_senLen` 目前未被使用，模型不在此檔主動截斷過長 prompt。
- `img_tok_id` 目前只保存，不直接參與此檔的 split 定位；split 依最後一個 SEP。
- `SYSTEM_PROMPT` 目前未使用。
- cached prefix 的正確性依賴 processor 保持「最後 SEP 後全為純文字」的序列布局。
- cached generation 只針對已列舉的 exception types fallback；其他 exception 直接向外傳遞。
- `eval_prompt_batch_size` 是 DxItem prompt batch 大小，不是 case batch 大小。
- `inputs_embeds` generation 對輸出 token 邊界的處理依賴目前 Transformers decoder-only API 行為。

這些邊界應在修改 processor、chat template、Transformers 版本或模型家族時重新驗證，但不應在未驗證下自行改變。


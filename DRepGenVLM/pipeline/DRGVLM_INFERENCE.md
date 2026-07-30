# DRGVLM inference pipeline

The pipeline generates case-level diagnostic text for metadata that uses the
same ROI and `DxPair` structure as `multiROI2DxResultDataset`.
`structured_report` is optional and is never used to build an inference
prompt.

Run from the DRepGenVLM project root:

```bash
python -m DRepGenVLM.pipeline.drgvlmInferencePipeline \
  --checkpoint-dir /path/to/DRGVLM_checkpoint \
  --input-metadata /path/to/input.json \
  --output-metadata /path/to/output.json
```

## Input contract

The metadata root must contain:

```json
{
  "DxItem_list": [
    "Histologic_Type",
    "Histologic_Grade",
    "Microcalcification"
  ],
  "case_list": []
}
```

Every case must have a unique `sample_idx`, a `case_id`, and at least one ROI
whose `DxPair` contains a declared DxItem. The ROI image fields are read from
the checkpoint config's `level_key`.

`structured_report` may be absent, empty, or `null`. When present it is
preserved unchanged.

## Output

All original metadata fields are preserved. Each case receives
`generated_report` at the same level as `structured_report`:

```json
{
  "sample_idx": 7,
  "case_id": "case-7",
  "structured_report": {},
  "generated_report": {
    "Histologic_Type": "Infiltrating duct carcinoma",
    "Histologic_Grade": "grade 2 (Nottingham histologic score: 6/9)",
    "Microcalcification": "Not identified"
  }
}
```

Only DxItems activated by that case's ROI `DxPair` keys are written. A
declared or structured-report DxItem without an assigned ROI is omitted.
Every expected generated value must be a non-empty string.

The input file is never overwritten. The output is written atomically.
An existing output file or a non-empty input `generated_report` causes the
pipeline to stop unless `--overwrite` is supplied.

## Checkpoint and sampling defaults

The default artifacts are:

- Config: `<checkpoint-dir>/config.json`
- Checkpoint: `<checkpoint-dir>/checkpoint_refs/best.json`

If the best reference is absent, the pipeline tries the legacy
`best_model.pth` layout. Use `--checkpoint-path` to select another checkpoint
reference, immutable snapshot directory, or legacy checkpoint request.

The following values are inherited from the saved checkpoint config:

- `max_rois_per_dxitem`
- `roi_sampling_mode`
- `valid_sampling_seed`
- `max_new_tokens`
- `eval_prompt_batch_size`
- `pp_num_gpus`
- `device`

They can be overridden explicitly:

```bash
python -m DRepGenVLM.pipeline.drgvlmInferencePipeline \
  --checkpoint-dir /path/to/checkpoint \
  --input-metadata /path/to/input.json \
  --output-metadata /path/to/output.json \
  --image-path /localized/image/root \
  --weight-path /localized/weight/root \
  --pp-num-gpus 4 \
  --max-new-tokens 256
```

Generation is deterministic (`do_sample=False`). `random_k` ROI sampling uses
the saved validation seed and a stable per-case/per-DxItem RNG.

## Preflight

Preflight validates metadata, output collisions, runtime paths, the best
checkpoint reference and snapshot hashes without loading the VLM:

```bash
python -m DRepGenVLM.pipeline.drgvlmInferencePipeline \
  --checkpoint-dir /path/to/checkpoint \
  --input-metadata /path/to/input.json \
  --preflight-only
```

It prints a JSON summary containing the case and active-DxItem counts,
checkpoint epoch, effective ROI assignment count, sampling settings, and
generation settings.

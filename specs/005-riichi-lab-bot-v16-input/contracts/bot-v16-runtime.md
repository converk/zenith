# Contract: Bot V16 Runtime

## Checkpoint Contract

- Required payload fields: `model_config`, `model`
- Required model config: `policy_head_type == "symmetric_action_query"`
- Loader behavior: instantiate `KyokuTransformerActorCritic(ModelConfig(**model_config))`, strict load `model`, move to selected device, eval mode
- Non-contract metadata: `sft_contract_version`, `ppo_format_version`, `token_schema_version`, optimizer/scheduler/RNG fields

## Prepared Input Contract

`PreparedDecision` supplies the exact keyword arguments for:

```python
model.forward_v16(
    history_factors,
    history_numeric,
    history_lengths,
    snapshot_kinds,
    snapshot_cat,
    snapshot_num,
    snapshot_lengths,
    query_rows,
    query_action_ids,
    query_pair_counts,
    legal_mask,
    policy_only=True,
)
```

Each array is a single-row batch at inference time. The bridge may store row arrays without the leading batch dimension, but `PolicyEngine` must add it before tensor conversion.

## Online Safety Contract

- The model-selected action id must be legal in `PreparedDecision.legal_mask`.
- Decoding is done by `MjaiKyokuStateMachineManager.decode_actions`.
- Final response must pass both `Observation.select_action_from_mjai()` and server `possible_actions`.
- If safety validation fails, fallback/withheld remains a failure signal in tests and telemetry.

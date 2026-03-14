# System2 Memory LoRA Seed Dataset

This synthetic seed set mirrors the `System2Session` contract used by `isaac-aura`.

## Contents

- image size: `320x320`
- seed: `7`
- train/val/test: `24/8/8`

## Task Families

- `direct_pixel`: 6
- `memory_pixel`: 6
- `memory_turn_left`: 7
- `memory_turn_right`: 7
- `stop`: 7
- `wait`: 7

## Schema

- `instruction`: Korean natural-language navigation command
- `decision_text`: expected System 2 output string
- `current_image` / `history_images`: image paths relative to the dataset root
- `memory_context`: serialized `MemoryContextBundle` compatible with `MemoryService.build_memory_context()`

Use this seed set for smoke tests and prompt-format stabilization, then replace or extend it with Isaac/real-robot captures using the same schema.

# pd-ocr-trainer: Deep Code Review

Reviewed 6 modules (~4,400 lines). Issues grouped by file in descending severity. Intended as a working document for iterative remediation.

---

## `train_recog.py` + `train_detect.py`

These two files are reviewed together because ~400 lines are near-verbatim duplicated between them. Every bug fixed in one must be fixed in both.

### Bugs

**1. `NameError: wandb_log_at_step` in distributed W&B runs**
`train_recog.py:688`, `train_detect.py:548`
`wandb_log_at_step` is defined only inside `if rank == 0 and args.wb:`. `log_at_step` is defined outside that block and passed to all ranks via `fit_one_epoch`. On non-rank-0 workers with `args.wb=True`, calling `log_at_step` triggers `NameError`. Hard crash in any distributed W&B run. `clearml_log_at_step` has the same defect.

**2. `RecognitionDataset` subsets in multi-part training paths have no `img_transforms`**
`train_recog.py:503`
When `train_path` is a multi-directory path, the first `RecognitionDataset` gets the full augmentation pipeline. Additional subsets added via `merge_dataset` (line 503) are constructed with no `img_transforms` at all. Samples from part 2 onward arrive at the collator at their native, unresized resolution, causing collation errors or silent non-augmented training data.

**3. Missing `sampler.set_epoch(epoch)` in distributed training**
`train_recog.py:556`, `train_detect.py:~474`
`DistributedSampler(shuffle=True)` requires `sampler.set_epoch(epoch)` at the start of each epoch to produce different shuffles per epoch. Without it, every epoch sees the same data ordering on every rank, defeating the purpose of shuffling in DDP runs.

**4. `record_lr` uses `.cuda()` instead of `.to(device)`**
`train_recog.py:106`, `train_detect.py:80`
`images.cuda()` hardcodes GPU 0 regardless of the configured device. On a multi-GPU machine training on cuda:1, the LR finder moves images to cuda:0, causing an immediate CUDA device mismatch error. `fit_one_epoch` correctly uses `images.to(device)` — the inconsistency is a copy-paste artifact.

**5. `optimizer`/`scheduler` are unbound for unrecognized strings in programmatic calls**
`train_recog.py:580-609`, `train_detect.py:487-514`
Both are assigned only inside `if`/`elif` chains. `parse_args` enforces `choices=`, but `train_from_config`/`detect_from_config` accept arbitrary strings. Passing `optimizer="sgd"` from the UI produces `UnboundLocalError` deep in the epoch loop rather than a clear error message. Add `else: raise ValueError(...)` after each chain.

**6. `weight_decay=0.0` treated as falsy**
`train_detect.py:501`
`args.weight_decay or 1e-4` evaluates to `1e-4` when `weight_decay=0.0`, silently ignoring an explicit zero-decay request. Fix: `args.weight_decay if args.weight_decay is not None else 1e-4`.

**7. `global global_step` as module-level mutable state**
`train_recog.py:633`, `train_detect.py:537`
The W&B/ClearML step counter is a module-level global reset at the top of `main()`. If `main()` is called twice in the same process (sequential training runs from the UI), the second call resets to 0 while the first run's logger may still be active. Fix: replace with `nonlocal` in the `log_at_step` closure.

**8. Checkpoint save is not atomic**
`train_recog.py:733`, `train_detect.py:618`
`torch.save()` writes directly to the target `.pt` file. A kill mid-write leaves a corrupt checkpoint. Sidecar files (`.vocab`, `.arch`) are in separate `try/except` blocks with no rollback if the `.pt` write already succeeded. Fix: write to `.tmp` then `Path.replace()` for atomicity.

**9. `class_names` not broadcast in DDP detection training**
`train_detect.py:360-367`
`class_names` is only set on rank 0 (inside `if rank == 0:`). On worker ranks it falls back to `["words"]`. The model is then instantiated with this hardcoded value on all non-rank-0 processes. For multi-class detection datasets the output head dimension diverges across ranks — a silent DDP architecture mismatch. Requires `dist.broadcast_object_list` before model construction.

**10. `test_only`/`find_lr`/`show_samples` early returns are rank-0-only**
`train_detect.py:388, 485, 504`
`if rank == 0: ... return` — worker ranks fall through and begin the full training loop. These modes are likely single-GPU in practice, but it is a correctness defect under any distributed invocation.

**11. Empty `DataLoader` crashes `fit_one_epoch` and `evaluate`**
Both files.
`last_lr` is only assigned inside the batch loop. An empty `DataLoader` (dataset smaller than `batch_size` with `drop_last=True`) causes `UnboundLocalError` on `last_lr` and `ZeroDivisionError` on the epoch-loss average. Neither function guards against zero batches.

### Dead Code

**`evaluate()` function exists in both files and is never called.**
`train_recog.py:216`, `train_detect.py:180`
Both files have both `evaluate` and `evaluate_with_progress`. Only the latter is called from `main`. `evaluate` is ~35 lines of duplicated dead code in each file. Delete it.

### Structural Issues

**~400 lines of near-verbatim duplication between the two trainers.**

| Block | Detection | Recognition | Status |
|---|---|---|---|
| `_emit_progress` / `ProgressHook` | 40–47 | 42–49 | Byte-for-byte identical |
| `record_lr` | 50–112 | 73–144 | ~95% identical |
| `fit_one_epoch` | 115–177 | 147–212 | ~95% identical |
| `evaluate` (dead) | 180–214 | 215–248 | ~90% identical |
| Optimizer construction | 487–502 | 580–595 | Byte-for-byte |
| Scheduler construction | 509–514 | 604–609 | Byte-for-byte |
| `global_step` / logger setup | 537–582 | 633–691 | ~90% identical |
| Checkpoint save + sidecars | 615–626 | 726–748 | ~90% identical |

All shared blocks belong in a new `train_utils.py` module. The metric-update logic (the only real difference) can be injected via a callable parameter or thin wrapper.

**`main()` is 400–500 lines in both files.** Natural splits:

- `_setup_device(args)` — device / DDP init
- `_build_val_loader(args, rank, ...)` — val dataset + loader
- `_build_train_loader(args, ...)` — train dataset + loader
- `_build_optimizer(model, args)` — optimizer construction
- `_build_scheduler(optimizer, args, steps)` — scheduler construction
- `_run_epoch_loop(...)` — the epoch for-loop
- Keep `main` as a thin coordinator.

**Naming inconsistencies:**

- `detect_from_config` vs `train_from_config` — one convention should be used.
- `args.optim` in the namespace vs `optimizer` in the public API.
- `evaluate` vs `evaluate_with_progress` implies meaningful difference; the only difference is a UI concern.

---

## `ui.py`

### Bugs

**12. Unbounded textarea growth**
`ui.py:1099`
`output_area.value +=` appends to the in-browser textarea string for every training event. For long runs (hundreds of epochs, batch-level callbacks), the DOM element grows without bound and will freeze the browser tab. Needs a ring-buffer or max-length truncation.

**13. Training state is process-wide, not per-session**
`ui.py:464-465, 1107`
`training_thread` and `training_cancelled` are module-level globals. In a multi-browser-session NiceGUI deployment, all sessions share the same training thread — session A can see session B's state, cancel it, or be blocked by it. Should be per-session state.

**14. `train_worker` closes over live config objects**
`ui.py:1186`
`train_worker` reads from the live `detection_config` and `recognition_config` globals at training time, not at click time. If the user edits a hyperparameter after clicking "Start", the training run silently uses the modified value. Config should be snapshotted as a frozen copy when `run_training` fires.

**15. `_detect_cuda_device()` is a phantom**
`ui.py:288, 318`
Both config classes have `self.device = None` with a comment referencing `_detect_cuda_device()`. That function does not exist anywhere in the codebase. The `.device` attribute is set in `train_worker` as a local variable and never written back. The attribute is never read. Stale comment, dead state.

**16. `refresh_vocab_ui()` can raise unhandled `ValueError`**
`ui.py:959`
`build_custom_vocab_arg()` raises `ValueError` on empty vocab. This call is unwrapped inside `refresh_vocab_ui()`, which fires on every keystroke in the custom-chars field. An empty vocab+library state propagates a traceback to the NiceGUI event loop with no user-facing error.

**17. Module-level side effects at import time**
`ui.py:459-465`
`migrate_legacy_dataset_layout()`, `ExportManager()`, two `TrainingConfig()` constructors, and `_load_trainer_settings()` all execute at `import pd_ocr_trainer.ui`. This triggers disk I/O and filesystem migration on every import, making the module untestable in isolation. Defer to `main()` or `create_ui()`.

**18. `_load_trainer_settings` swallows all exceptions silently**
`ui.py:449`
`except Exception: pass` — any failure (corrupt file, permissions) is invisible to the user. Add `logger.warning(...)`.

**19. Bool loading uses `bool(string)` not `== True`**
`ui.py:389-392, 407-409`
`bool(data.get("rotation", ...))` is correct for JSON-loaded booleans, but a hand-edited `"false"` string evaluates to `True`. Use `data.get("rotation") is True` or an explicit comparison.

**20. Default model timestamps are hour-granularity**
`ui.py:264`
Two training runs within the same clock hour produce the same default model name, silently overwriting the first run's output. Use `%Y%m%d%H%M` (minute-granularity).

**21. `label_count` JSON open has no `encoding="utf-8"`**
`ui.py:1118`
Every other `json.load` call in the file specifies `encoding="utf-8"`. This one does not. Fails on non-UTF-8 locale systems with non-ASCII characters in labels.

### Structural Issues

**`create_ui()` is 860 lines** — the dominant structural problem. It contains profile card, detection card, recognition card, and output card construction plus six nested closures and the training thread machinery. Natural splits:

- `_build_profile_card(...)` — lines 498–544
- `_build_detection_card(...)` — lines 557–760
- `_build_recognition_card(...)` — lines 761–1070
- `_build_output_card(...)` — lines 1072–1316

The shared `output_labels` dict and `_kanban_refresh` list are already mutable containers passed by reference — pre-create at the top of `create_ui` and pass down.

**`_safe_int`/`_safe_float` are pure functions defined inside `create_ui`** (`ui.py:482, 490`). No closure dependency. Should be module-level.

**`label_count` is a pure utility nested three levels deep** (`ui.py:1113`). Should be module-level.

**`DetectionTrainingConfig` and `RecognitionTrainingConfig` are plain-old-data with no validation** (`ui.py:270, 294`). Promote to `@dataclass` with `__post_init__` validation.

**Config serialisation/deserialisation are mirrored field-by-field.** `_detection_settings_payload` + `_apply_detection_settings`, and `_recognition_settings_payload` + `_apply_recognition_settings`. Adding any field requires edits in two places. Use `dataclasses.asdict` or a `__dict__`-based approach.

**`_kanban_refresh` as a single-element list** (`ui.py:480`) is a workaround that should be replaced with `nonlocal`.

**`refresh_model_output_labels()` conflates three responsibilities** (`ui.py:505`): updating path display labels, updating a scope label, and calling `export_manager.set_profile()` + `export_manager.scan()`. The profile-change side effect belongs in a dedicated `on_profile_change` handler.

---

## `dataset_store.py`

### Bugs

**22. Non-atomic `labels.json` writes throughout**
`dataset_store.py:407` and all `open(..., "w")` calls in `move_existing_project` (lines 460–463), `move_existing_page` (lines 508–511), and `migrate_legacy_dataset_layout`.
`_write_json_map` opens the file for write (truncating it immediately) then calls `json.dump`. A crash mid-write leaves a permanently empty or corrupt `labels.json`. Fix everywhere: write to `.tmp` then `Path.replace()` (atomic on POSIX).

**23. Module-level `mkdir` at import time**
`dataset_store.py:50-54`
Five `Path.mkdir(exist_ok=True)` calls execute at import. In a read-only filesystem, containerized test environment, or any import from a static analysis tool, this raises `OSError` or silently creates directories in unexpected locations. Defer to a `setup()` or `ensure_dirs()` function called from `main()`.

**24. `os.getenv` default is a `Path` object**
`dataset_store.py:19-20`
`os.getenv("PD_OCR_TRAINER_ML_TRAINING_DIR", PROJECT_ROOT / "ml-training")` — `os.getenv`'s default parameter is typed `str | None`. Passing a `Path` breaks static type checkers. Fix: `str(PROJECT_ROOT / "ml-training")`.

**25. `iter_export_profile_dirs` uses `rglob("*")` instead of `iterdir()`**
`dataset_store.py:97`
`rglob("*")` descends recursively into `images/` subdirectories, checking every file for `labels.json`. For large image trees this is extremely slow. The structure is known to be one level deep. Use `project_dir.iterdir()`.

**26. `move_existing_project`/`move_existing_page` bypass `_load_json_map`/`_write_json_map`**
`dataset_store.py:434-463, 479-511`
These methods contain raw `open()` + `json.load()` / `json.dump()` instead of reusing the class's own helpers. This is the reason those paths lack atomic write protection and is a clear DRY violation.

**27. `save_assignments` count semantics are wrong**
`dataset_store.py:576`
`count += 1` increments per `(key, task)` pair, not per page. With both detection and recognition tasks, every key increments count by 2. The returned `{"copied": count}` is displayed to users but is double the number of export directories processed.

**28. No file locking on `labels.json` read-modify-write**
Throughout `dataset_store.py`.
Concurrent processes (two UI tabs, a training script alongside the UI) can corrupt label files. `fcntl.flock` or the `filelock` library should guard every read-modify-write cycle.

**29. Private class methods used from module-level functions**
`dataset_store.py:151, 220`
`group_existing_by_project` and `migrate_legacy_dataset_layout` reach into `ExportManager._load_json_map` and `ExportManager._write_json_map`. These helpers have no class state dependency. Make them module-level functions.

### Structural Issues

**`save_assignments` plan-build and execution are interleaved** (lines 514–587, ~74 lines). Separate `_build_copy_plan()` and `_execute_copy_plan()` methods would make each independently testable.

**`move_existing_project` and `move_existing_page` are ~60 lines of near-identical logic.** A shared `_move_items(filter_fn, from_split, to_split)` helper would unify them.

**`_drop_page_assignments_for_key(key)` pattern appears 3 times** (`lines 343, 364, 581`). Extract to a method.

**Split-validation guard** (`if split not in {"train", "val"}: return`) appears in six methods in slightly different forms. Extract to `_validate_split(split) -> bool`.

**`assign()` and `assign_project()` rebuild the full `page_assignments` dict on every call** (`lines 343, 364`). O(N) per call. For a project with K keys this is O(K×N). Use a per-key index or a simple `del` loop.

**`ExportManager.split_root()` creates directories as a side effect** — surprising in a read path.

---

## `dataset_ui.py`

### Bugs

**30. `model_profile_state` parameter is never read**
`dataset_ui.py:20`
The parameter is declared but no code in the function body reads it. Either remove it or document why it is passed.

**31. "Copy to Datasets" reports zero even with page-level assignments**
`dataset_ui.py:298`
`save_assignments` returns `{"copied": count}` where count is task-pairs. When page-level assignments exist but the key has no full-split assignment, `save_assignments` can return 0, triggering a false "No assigned exports to copy" warning.

**32. `handle_drop` gives no feedback for on-disk moves**
`dataset_ui.py:99`
After a successful `move_existing_page` or `move_existing_project` call, `refresh_kanban()` fires but `kanban_status.set_text(...)` is not called. Users get no confirmation the move succeeded.

**33. O(N+1) JSON reads per column per `refresh_kanban` call**
`dataset_ui.py:144, 206`
`render_column` calls `get_existing_projects(split_root)` (parses `labels.json`), then for each of N projects calls `get_existing_pages(split_root, project_id)` which re-parses `labels.json`. Three columns × (N+1) reads per refresh. Cache the result of `group_existing_by_project` for the duration of one `render_column` call.

### Structural Issues

**`build_dataset_section` is 300 lines with five nested closures** — `_select_existing_page`, `handle_drop`, `render_column`, and `save_assignments` are meaningful units with no inseparable closure dependency. Extract to module-level functions or a `KanbanState` class to make them independently testable.

**Shared mutable closure state** (`dragging`, `selected_existing_pages`, `page_selection_anchor`) is not thread-safe across NiceGUI sessions. Add a comment warning about single-session assumption.

---

## `utils.py`

### Bugs

**34. `plot_samples` crashes with 1–3 images**
`utils.py:19-26`
With 2 or 3 images, `plt.subplots(1, N)` returns a 1D `axes` array. The code does `ax = axes[row_idx]` (returns an `Axes`) then `ax = ax[col_idx]` (subscripts an `Axes` object → `TypeError`). With exactly 1 image, `axes.ravel()` at line 26 raises `AttributeError` because `plt.subplots(1, 1)` returns a bare `Axes`, not an ndarray. Fix: use `squeeze=False` in `plt.subplots` and always index with `axes[row_idx, col_idx]`.

**35. Dead code in `plot_recorder`**
`utils.py:59`
`np.ndarray.argmin()` never returns `None`. The branch `if min_idx is None` is unreachable. Remove it.

**36. Neither plotting function calls `plt.close()`**
`utils.py:30, 68`
Figure objects accumulate across calls. In a training loop this will hit matplotlib's figure limit. Add `plt.close()` after `plt.show()` in both functions.

**37. `suggest_batch_size` belongs in `utils.py`, not `ui.py`**
`ui.py:169`
It has no NiceGUI dependency and would be independently testable here. Currently `utils.py` is never imported by `ui.py`.

---

## `tests/`

### Correctness Issues

**38. Session-scoped fixture + mutable on-disk state = order-dependent tests**
`conftest.py:23`
`browser_dataset_root` is `scope="session"`. `test_drag_single_page.py` moves a page from training to validation, changing on-disk state for all subsequent tests in the session. Use `scope="function"` with `tmp_path` or add explicit reset logic in each test.

**39. `plot_samples` crashes are completely untested**
The axes-indexing bugs (issue 34) are fully unit-testable but there are no tests for `plot_samples`.

**40. Fragile CSS-class locators in browser tests**
`test_drag_single_page.py:26`, `test_drag_export_single_page.py:28`
Both use `.min-h-16` as a Playwright locator. This Tailwind utility class can be renamed or stripped. Prefer `data-testid` attributes on drop zones.

**41. Subprocess stdout pipe buffer is never drained**
`conftest.py:81-88`
The NiceGUI subprocess's stdout is captured but never read. On Linux the pipe buffer is ~64 KB. A long-running test session will cause the subprocess to block writing to stdout, hanging the test. Drain in a background thread.

**42. No unit tests for `dataset_ui.py` event handlers**
`_select_existing_page` (range selection, anchor logic), `handle_drop` (five dispatch paths), and `render_column` contain non-trivial logic with zero unit test coverage.

**43. No test for "Copy to Datasets" + kanban refresh integration.**

**44. `EarlyStopper` plateau case not tested**
`test_utils.py:13-18`
A loss that stays exactly at `min_validation_loss` never increments the counter regardless of patience. Not tested. Also untested: counter reset mid-degradation, `patience=1`, `min_delta=0.0`.

---

## Priority Matrix

| # | Issue | File | Severity |
|---|---|---|---|
| 22 | Non-atomic `labels.json` writes | `dataset_store.py` throughout | Critical |
| 2 | `RecognitionDataset` subsets lack `img_transforms` | `train_recog.py:503` | Critical |
| 1 | `NameError` in distributed W&B | `train_recog.py:688`, `train_detect.py:548` | Critical |
| 34 | `plot_samples` crashes with ≤3 images | `utils.py:19-26` | Critical |
| 3 | Missing `sampler.set_epoch()` in DDP | both trainers | High |
| 9 | `class_names` not broadcast in DDP | `train_detect.py:360` | High |
| 12 | Unbounded textarea growth | `ui.py:1099` | High |
| 13 | Training state is process-wide | `ui.py:464-465` | High |
| 17 | Module-level I/O side effects at import | `ui.py:459-465`, `dataset_store.py:50-54` | High |
| 4 | `record_lr` uses `.cuda()` not `.to(device)` | both trainers | Medium |
| 5 | `optimizer`/`scheduler` unbound for bad strings | both trainers | Medium |
| 8 | Non-atomic checkpoint save | both trainers | Medium |
| 14 | `train_worker` closes over live config | `ui.py:1186` | Medium |
| 15 | `_detect_cuda_device()` phantom | `ui.py:288, 318` | Medium |
| 16 | `refresh_vocab_ui` can raise unhandled `ValueError` | `ui.py:959` | Medium |
| 7 | `global global_step` module-level mutable | both trainers | Medium |
| dead | `evaluate()` dead in both trainers | both trainers | Medium |
| dup | ~400 lines duplicated across trainers | both trainers | Medium |
| 6 | `weight_decay=0.0` treated as falsy | `train_detect.py:501` | Low |
| 20 | Hour-granularity default model timestamps | `ui.py:264` | Low |
| 38 | Session-scoped mutable fixture | `conftest.py:23` | Low |
| 33 | O(N+1) JSON reads per kanban refresh | `dataset_ui.py:144, 206` | Low |

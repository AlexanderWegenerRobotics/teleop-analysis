# teleop-analysis reference

The full details behind the [README](../README.md).

## Using it

- Left: one tree per opened folder, mirroring the folder structure. Expand a file to see its signals; numbered columns (`q_0..q_6`, `O_T_EE_0..15`) are grouped. The search box matches all words, e.g. `005 arm_left tau_ext`.
- Drag a signal or a whole group onto a plot. Drop it on the strip at the bottom to get a new panel. Double-click adds it to the highlighted panel.
- Event files (`events.log`, `*_meta.csv`): double-click to show them as markers and in the Events list. Click an event to jump there.
- Video files (`*.timestamps.csv` with a matching `.h264`/`.mp4`/`.hdf5`, `frames.jsonl` with `session.mp4`): double-click to open. The frame shown is the last one captured at or before the cursor. Drag the slider under the video (or the ◀ ▶ buttons) to scrub; the cursor in all plots follows, so you can find the moment in the video first and then look at the numbers.
- Mouse: left drag pans, right drag draws a box and zooms into it (x and y), wheel zooms time, Shift+wheel (or wheel over the y axis) zooms y, click sets cursor A. Double-click fits all, Backspace goes back to the previous zoom. Right click without dragging still opens the menu. The mouse pointer shows what a drag will do: open/closed hand = pan, ↔ = time cursor or panel width, ↕ = y cursor, panel border or y axis, pointing hand = over a curve.
- Hover a curve to see its value at that point. Right-click it: rename, colour, line width, line style, hide, move to another panel, math, remove.
- "New panel" adds a panel below. Its arrow opens a picker with panel arrangements (stacked, side by side, 2 × 2, 3 × 2, one big + two small, …). Picking one re-arranges the current page and keeps its plots in order, or, with "open as a new page", starts a new page with that arrangement. All panels of a page share the time axis; the split lines can be dragged.
- Pages: tabs above the plots. "+" (Ctrl+T) opens a new page, double-click a tab to rename it, right-click to duplicate or close. Each page keeps its own panels and zoom; the cursors and event markers are shared. Saved views contain all pages; export writes the current page.
- Moving signals between panels: drag rows from the Values panel onto any plot (or the strip at the bottom for a new panel), or right-click → "Move to panel". Panels that end up empty are removed.
- Ctrl+click curves (or select rows in the Values panel) to select several; right-click one of them, or empty plot area → "Math on selected signals". Esc clears the selection.
- XY plot: Ctrl+click two curves (first = x, second = y), right-click → "XY plot". A floating window shows y over x for the visible time window (or the whole log), with cursor A marked; "swap x/y" flips the axes. Use this for e.g. gaze pixel u over v, or force over displacement.
- "Cursors here" (C) moves cursor A, and B if it is on, into the visible window without changing the zoom.
- Δ cursor (D, or Shift+click): second time cursor B. The plot shows Δt and 1/Δt, the Values panel shows each signal at A, at B and the difference.
- Right-click empty plot area → "Y cursors (Δy)": two horizontal lines showing y₁, y₂ and Δy.
- ▶ / ❚❚ (Space) plays from the cursor. ◀◀ ▶▶ (Left/Right) step by the size chosen next to them: one video frame, one sample of the selected (or first) signal, or a fixed 1 ms … 1 s.
- Values panel: value of every signal at the cursor(s). Double-click a name to rename it (used in legend and export), the colour square to change colour.
- Ctrl+E exports the current window. Default is just a png. Optional: pdf/svg, cursor lines, the video frame(s) at cursor A (next to the plots or as separate images), the data as CSV and the view file. The CSV holds every visible signal of every panel, only the visible time window, each at its own sample times (`_data.csv` long format, `_wide.csv` one column per signal). The last used options are remembered.
- Ctrl+S / Ctrl+L save and load views (derived signals included).

## Intent posteriors and the video overlay

The `modules/intent/<arm>` streams of `local.hdf5` show readable names instead of indices: `phase_posterior_k` as `phase approach`, `phase insert`, … (orchestrator `Phase` order) and `target_posterior_k` / `location_posterior_k` as `target peg_1_2`, `target goal corridor`, …, with empty candidate slots greyed out. Phase curves get the label tool's phase colours. Search works on these names too (`intent left target peg_1`).

Where the candidate names come from, in this order (shown in the stream's tooltip):

1. `frames/candidate_names` + the `phase_names` attribute, which the orchestrator's RunLogger writes from now on.
2. Older logs: objects matched by position to `scene.csv` in the same episode folder, goals to the task yaml the run's orchestrator config points at (`geometry.goals_from`; the orchestrator repo is found next to teleop-analysis, or set `TELEOP_ORCHESTRATOR`).
3. Otherwise `object k` / `goal k`.

Videos in an episode that has a `local.hdf5` get an "Intent" button. It writes each arm's top phase and top target (with probability) in the corner, and, for a static camera with an entry in `camera_params.json`, draws every candidate's name at its projected position and rings the predicted target in the arm's colour (left cyan, right orange). `camera_params.json` is looked up in the episode folder and its parents, then in the avatar log root from the default paths; "Collect episodes" now copies it along. The interface videos (`session*.mp4`) have no camera model, so they get the text only. Export with "video frame" includes the overlay when it is on.

The "Gaze" button next to it draws the operator's gaze (ring = latest sample, red trail = last 0.4 s, hidden when gaze is invalid or older than 0.25 s). On the desk camera it uses `gaze_px_x/y` from `intention_log.csv` (gaze rays, mapped to pixels with the camera's fx/fy/cx/cy from `camera_params.json`); on the interface videos it uses `gaze_u/v` from `frames.jsonl`, the same values `session_attention.mp4` is rendered from. Both buttons remember their state for new video windows.

## Math

Available from the right-click menus in the plots, the Values panel and the tree. Results show up under "Derived" in the tree, behave like any other signal (plot, export, math on math), and are recomputed when clocks or shifts change.

| | inputs | result |
|---|---|---|
| SE(3) → position + RPY + quaternion | 16 (`O_T_EE_0..15`, column- or row-major detected) | x y z [m], roll pitch yaw [deg, ZYX, unwrapped], qw qx qy qz. A second pose (e.g. `O_T_EE_cmd`) goes into the same position / orientation panels, same colour per axis, dashed |
| Quaternion → roll / pitch / yaw | 4 (w, x, y, z; order from names) | roll pitch yaw [deg] |
| Magnitude, sum | ≥ 1 / ≥ 2 | ‖·‖, a + b + … |
| Difference, product, ratio | 2 | a − b, a · b, a / b |
| Expression | ≥ 1 | any numpy expression in a, b, c, …, t |
| Derivative, integral, moving average, abs, scale/offset, time shift, unwrap, rad↔deg | each input separately | one signal per input |
| Estimate delay | 2 | cross-correlation lag of b vs a in the visible window; optionally adds b shifted back |

Inputs with different sample times are interpolated onto the first input's timestamps; gaps stay gaps.

## Time

Everything is drawn on the interface / operator-PC clock, in seconds from an origin (Toolbar: "t = 0 at cursor").

| data | time column | clock |
|---|---|---|
| avatar CSVs (`arm_*`, `head`, `scene`, `intention_log`) | `wall_clock_ns` / `timestamp_ns` | avatar (NUC) |
| `*_wrench_truth.csv` | `sim_time`, mapped through `sim_time`/`wall_clock_ns` of the sibling arm log | avatar |
| `*_meta.csv` | `time_s` from the start of the sibling data file | avatar |
| `*.timestamps.csv` | `capture_time_ns`, else `wall_clock_ns` | avatar, operator if in an `operator` folder |
| `command.csv`, `stream.csv`, `events.log`, `frames.jsonl` | `timestamp_ns` / `ts` | interface |
| `local.hdf5` (local orchestrator: `frames/*`, `modules/**` as one stream per group) | `frames/timestamp_ns` | avatar (scene tick stamps) |
| `nudge.csv` (teleop-nudge event log; issued cues also as events) | `t_ns` | avatar |
| ROS2 `.db3` | `header.stamp`, else bag receive time | by folder name |
| any other CSV | first epoch-looking `*_ns` column, else `time`/`t` | guessed from folder name |

The avatar offset is estimated per opened folder from `command.csv` (same method as `sync_triptych.py`: one-way = (median rtt − 9 ms)/2). "Clocks…" shows the estimate and lets you override it. Right-click a folder → "Align folder start to t = 0" or a manual shift to overlay episodes recorded at different times.

Other columns ending in `_ns` become `<name> - t [ms]` (epoch stamps, relative to the row time) or `<name> [ms]` (durations). Timestamp files get `frame_dt [ms]`.

## Missing and new data

- Files are recognised by name, so a missing file just means that stream is not there. Header-only files show as `(empty)`; files nothing can read are listed under "not loaded".
- Gaps longer than 10× the median sample period (min 0.25 s) are drawn as gaps, not interpolated.
- Truncated last rows and broken timestamps are dropped.
- F5 rescans; new files show up, changed files are re-indexed.

## Cache

The first time a file is plotted it is converted to Parquet, which makes later loads instant. The cache is in `%LOCALAPPDATA%\teleop-analysis\cache` (or `TELEOP_ANALYSIS_CACHE`) and is keyed on path, size and modification time, so the log folders are never written to. Raw `.h264` is remuxed (no re-encode) into the same cache once so it can be seeked.

## Default paths

File → "Default paths…" sets the avatar, local orchestrator and interface log roots, the dataset folder and the `teleop-orchestrator` repo for this machine. They are saved outside the repo, in `%APPDATA%\teleop-analysis\paths.json` (Linux: `~/.config/teleop-analysis/paths.json`, or `TELEOP_ANALYSIS_CONFIG`). An old `teleop_analysis/collect_defaults.json` is still read as a fallback and is git-ignored.

## Collecting an episode

Every producer logs under the avatar's `session_id` / episode index (avatar `EpisodeEventMsg` fan-out): the avatar in `<log_dir>/<NNN>/`, the local orchestrator in `logs/<session_id>/<NNN>/` (`local.hdf5`, `nudge.csv`), the interface in `Saved/Logs/TeleOp/<session_id>/<NNN>/`. File → "Collect episodes…" in the viewer does this from a dialog. Any field accepts the log root, a session folder or an episode folder (the session and episode are read from the path); the other roots are guessed from sibling repos or taken from the default paths (see below). Empty episodes = the whole session. The dataset folder can be opened right after. `teleop-nudge/scripts/local_session.ps1` starts sim + local orchestrator + interface and runs the collection when you stop them. The same thing from the terminal:

```
teleop-collect 1791328557 3 --out D:\data\fmb-1-desk ^
    --avatar C:\...\teleop-simulator\logs --local C:\...\teleop-orchestrator\logs ^
    --interface C:\...\Windows\teleop_vr_interface\Saved\Logs\TeleOp
teleop-collect 1791328557 3 4 5 --out ... --avatar ... --with-video --attention
```

The avatar files, `local.hdf5` and `nudge.csv` land in `<out>/<NNN>/`, the interface files in `<out>/<NNN>/interface/`. Videos (avatar `.h264`, interface `session*.mp4`) are only copied with `--with-video`; `--attention` renders `session_attention.mp4` when the interface did not write one. `teleop-attention <interface episode dir>` does the same rendering on its own (replaces `teleop_vr_interface/Tests/render_attention.py`).

## From a notebook

```python
from teleop_analysis import open_folder
cat = open_folder(r"D:\data\DAgger")
cat.find("005 arm_left tau_ext")
s = cat.series("DAgger/005/arm_left.csv::tau_ext_3")
df = cat.frame("DAgger/005/head.csv")
```

## New log types

Add a loader in `teleop_analysis/loaders/` (subclass `CsvLoader` and set `prefer_time`, or `Loader` with `match`/`discover`/`build`) and put it in `LOADERS`, or call `teleop_analysis.register(MyLoader())`. Bump its `version` when the parsing changes so the cache is rebuilt.

## Tests

```
pip install pytest
pytest tests
```

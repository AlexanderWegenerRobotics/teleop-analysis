import json

import numpy as np
import pytest

from teleop_analysis import open_folder
from teleop_analysis.model import VIDEO

T0 = 1790625733_000_000_000


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("TELEOP_ANALYSIS_CACHE", str(tmp_path_factory.mktemp("cache")))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def session(tmp_path):
    av = tmp_path / "ep" / "avatar"
    rows = ["time;wall_clock_ns;sim_time;q_0;state"]
    for i in range(1000):
        if 400 <= i < 600:
            continue
        rows.append(f"{10 + i * 0.002:.6f};{T0 + i * 2_000_000};{5 + i * 0.002:.6f};{np.sin(i / 100):.6f};4")
    write(av / "arm_left.csv", "\n".join(rows) + "\n1.0;17906257")
    write(av / "arm_left_meta.csv", "session_id;episode_id;event;time_s;seed;mode;color_bin_mapping\n"
                                    "1;0;episode_start;0.0;;;\n1;1;episode_end;1.5;;;operator_idle\n")
    wr = ["sim_time;device;F_world_fz"] + [f"{5 + i * 0.005:.4f};arm_left;{i}" for i in range(300)]
    write(av / "arm_left_wrench_truth.csv", "\n".join(wr))
    write(av / "head.csv", "time;wall_clock_ns;q_0\n")
    write(av / "video_cam.timestamps.csv", "frame_idx,wall_clock_ns,capture_time_ns\n"
          + "\n".join(f"{i},{T0 + i * 33_000_000 + 5_000_000},{T0 + i * 33_000_000}" for i in range(30)))
    it = tmp_path / "ep" / "interface"
    cmd = ["timestamp_ns;device_index;px;rtt_ms;robot_recv_ns;robot_tx_ns"]
    for i in range(400):
        for d in (0, 1):
            t = T0 + 50_000_000 + i * 8_000_000
            cmd.append(f"{t};{d};{d + i * 0.001};30.0;{t + 1_000_000};{t - 20_000_000}")
    write(it / "command.csv", "\n".join(cmd))
    write(it / "events.log", f"{T0 + 100_000_000}  SESSION_START utc=x\n{T0 + 900_000_000}  GRASP side=left state=held\n"
                             "garbage line\n")
    frames = "\n".join(json.dumps({"frame_idx": i, "ts": (T0 + i * 33_000_000) / 1e9, "gaze_u": 0.5,
                                   "sender_time_ns": float(T0 + i * 33_000_000 - 2_000_000)}, indent=1)
                       for i in range(20))
    write(it / "frames.jsonl", frames)
    write(it / "notes.dat", "x")
    write(tmp_path / "ep" / "broken.csv", "")
    return tmp_path


def test_discovery_and_signals(session):
    cat = open_folder(session)
    root = cat.roots[0]
    keys = {s.key for s in root.streams}
    name = root.name
    assert f"{name}/ep/interface/command.csv#device_index=0" in keys
    assert f"{name}/ep/interface/command.csv#device_index=1" in keys
    assert "ep/interface/notes.dat" in root.skipped
    head = cat.stream(f"{name}/ep/avatar/head.csv")
    assert head.empty

    t, y = cat.get(f"{name}/ep/avatar/arm_left.csv::q_0")
    assert len(t) == 800
    assert np.all(np.diff(t) >= 0)

    t1, y1 = cat.get(f"{name}/ep/interface/command.csv#device_index=1::px")
    assert np.all(y1 >= 1)

    tw, _ = cat.get(f"{name}/ep/avatar/arm_left_wrench_truth.csv::F_world_fz")
    ta, _ = cat.get(f"{name}/ep/avatar/arm_left.csv::q_0")
    assert abs(tw[0] - ta[0]) < 0.01

    te, kinds, labels = cat.events(f"{name}/ep/interface/events.log")
    assert list(kinds) == ["SESSION_START", "GRASP"]
    tm, mk, ml = cat.events(f"{name}/ep/avatar/arm_left_meta.csv")
    assert abs((tm[1] - tm[0]) - 1.5) < 1e-6

    v = cat.stream(f"{name}/ep/avatar/video_cam.timestamps.csv")
    assert v.kind != VIDEO
    tv, fd = cat.get(f"{name}/ep/avatar/video_cam.timestamps.csv::frame_dt [ms]")
    assert np.nanmedian(fd) == pytest.approx(33.0)

    tf, gu = cat.get(f"{name}/ep/interface/frames.jsonl::gaze_u")
    assert len(tf) == 20


def test_clock_offset_estimate(session):
    cat = open_folder(session)
    root = cat.roots[0]
    est = cat.estimate_offset_ms(root, "avatar")
    assert est == pytest.approx(-((30.0 - 9.0) / 2 - 21.0))


def test_new_file_is_picked_up(session):
    cat = open_folder(session)
    write(session / "ep" / "avatar" / "extra.csv", "wall_clock_ns;x\n" + f"{T0};1\n{T0 + 1000};2\n")
    cat.reload()
    t, y = cat.get(f"{cat.roots[0].name}/ep/avatar/extra.csv::x")
    assert list(y) == [1, 2]


def test_export(session, tmp_path):
    from teleop_analysis.export import export_view
    cat = open_folder(session)
    name = cat.roots[0].name
    view = {"x_range": [0.0, 1.0], "panels": [{"signals": [{"key": f"{name}/ep/avatar/arm_left.csv::q_0",
                                                             "label": "q0", "color": "#1f77b4"}]}],
            "event_list": [[0.5, "GRASP", "GRASP held"]]}
    out = export_view(cat, view, tmp_path / "out" / "fig", formats=("png",), data=True, view_json=True)
    names = {p.name for p in out}
    assert {"fig.png", "fig_data.csv", "fig_wide.csv", "fig_view.json"} <= names

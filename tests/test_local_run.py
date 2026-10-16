import csv

import h5py
import numpy as np
import pytest

from teleop_analysis import open_folder
from teleop_analysis.collect import collect_episode

T0 = 1790625733_000_000_000


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("TELEOP_ANALYSIS_CACHE", str(tmp_path_factory.mktemp("cache")))


def make_local(folder):
    folder.mkdir(parents=True, exist_ok=True)
    n = 50
    with h5py.File(folder / "orchestrator.hdf5", "w") as f:
        f.create_dataset("frames/timestamp_ns", data=T0 + np.arange(n) * 10_000_000)
        f.create_dataset("frames/engaged", data=np.ones(n, dtype=bool))
        f.create_dataset("frames/candidate_mask", data=np.zeros((n, 3), dtype=bool))
        f.create_dataset("modules/intent/left/phase_posterior", data=np.random.rand(n, 7))
        f.create_dataset("modules/nudge/n_issued", data=np.zeros(n))
    with open(folder / "nudge.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_ns", "tick", "nudge", "tier", "arm", "strength", "urgency", "outcome",
                                          "channel", "primitive", "intensity", "phase_left", "phase_right",
                                          "entropy_left", "entropy_right", "state_source", "payload"])
        w.writeheader()
        for i in range(10):
            w.writerow({"t_ns": T0 + i * 100_000_000, "tick": i, "nudge": "contact_onset", "tier": "reflexive",
                        "arm": "left", "strength": 0.5, "urgency": 1, "outcome": "issued" if i % 2 else "suppressed:budget",
                        "channel": "haptic_left", "primitive": "pulse", "intensity": 0.8, "phase_left": "reach",
                        "phase_right": "idle", "entropy_left": 0.3, "entropy_right": 1.2, "state_source": "local_sim",
                        "payload": "{}"})


def test_local_hdf5_and_nudge(tmp_path):
    make_local(tmp_path / "1790000000" / "003")
    cat = open_folder(str(tmp_path))
    subs = {(s.rel, s.sub, s.kind) for s in cat.streams()}
    assert ("1790000000/003/orchestrator.hdf5", "frames", "signals") in subs
    assert ("1790000000/003/orchestrator.hdf5", "modules/intent/left", "signals") in subs
    assert ("1790000000/003/nudge.csv", "issued", "events") in subs
    key = [k for k in cat.find("phase_posterior_3")][0]
    assert len(cat.series(key)) == 50
    assert len(cat.series(cat.find("candidate_mask_2")[0])) == 50
    t, types, labels = cat.events(cat.find("nudge.csv issued")[0])
    assert len(t) == 5 and set(types) == {"contact_onset"}
    assert len(cat.series(cat.find("nudge.csv intensity")[0])) == 10


def test_collect_episode(tmp_path):
    av = tmp_path / "avatar" / "003"
    av.mkdir(parents=True)
    (av / "arm_left.csv").write_text("wall_clock_ns;q_0\n1;0\n")
    (av / "arm_left_meta.csv").write_text("session_id;episode_id;event;time_s\n1790000000;0;episode_start;0\n")
    (av / "video_desk_cam.h264").write_bytes(b"x")
    make_local(tmp_path / "local" / "1790000000" / "003")
    it = tmp_path / "iface" / "1790000000" / "003"
    it.mkdir(parents=True)
    for n in ("stream.csv", "command.csv", "events.log", "frames.jsonl", "session.mp4"):
        (it / n).write_text("")
    dst, copied, warnings = collect_episode("1790000000", 3, tmp_path / "out", tmp_path / "avatar",
                                            tmp_path / "local", tmp_path / "iface")
    assert dst == tmp_path / "out" / "003" and not warnings
    assert "video_desk_cam.h264" not in copied and "interface/session.mp4" not in copied
    assert {"arm_left.csv", "orchestrator.hdf5", "nudge.csv", "interface/stream.csv"} <= set(copied)
    _, copied, warnings = collect_episode("1790000001", 3, tmp_path / "out2", tmp_path / "avatar", None, None, with_video=True)
    assert "video_desk_cam.h264" in copied and any("belongs to session" in w for w in warnings)


def test_intent_labels_logged_and_fallback(tmp_path):
    import h5py
    ep = tmp_path / "005"
    ep.mkdir()
    n = 20
    types = np.tile([0, 0, -1, 2], (n, 1))
    mask = np.tile([True, True, False, True], (n, 1))
    pos = np.tile(np.array([[0.6, 0.15, 0.7], [0.6, -0.15, 0.7], [np.nan] * 3, [0.8, 0.0, 0.705]]), (n, 1, 1))
    with h5py.File(ep / "orchestrator.hdf5", "w") as f:
        f.create_dataset("frames/timestamp_ns", data=T0 + np.arange(n) * 10_000_000)
        f.create_dataset("frames/candidate_types", data=types)
        f.create_dataset("frames/candidate_mask", data=mask)
        f.create_dataset("frames/candidate_world_pos", data=pos)
        f.create_dataset("modules/intent/left/phase_posterior", data=np.random.rand(n, 7))
        f.create_dataset("modules/intent/left/target_posterior", data=np.random.rand(n, 4))
    (ep / "scene.csv").write_text("time;obj0_name;obj0_x;obj0_y;obj0_z;obj1_name;obj1_x;obj1_y;obj1_z\n"
                                  "0;peg_b;0.6;-0.15;0.71;peg_a;0.6;0.15;0.71\n")
    cat = open_folder(str(tmp_path))
    s = [s for s in cat.streams() if s.sub == "modules/intent/left"][0]
    lab = s.info["col_labels"]
    assert lab["phase_posterior_1"] == "phase approach" and lab["phase_posterior_5"] == "phase insert"
    assert lab["target_posterior_0"] == "target peg_a" and lab["target_posterior_1"] == "target peg_b"
    assert lab["target_posterior_2"].endswith("(empty)") and lab["target_posterior_3"] == "target goal 0"
    assert cat.find("target peg_b")
    with h5py.File(ep / "orchestrator.hdf5", "a") as f:
        f.create_dataset("frames/candidate_names", data=np.array([["x", "y", "", "corridor"]] * n, dtype=object)
                         .astype(h5py.string_dtype()))
        f.attrs["phase_names"] = np.array(["a", "b", "c", "d", "e", "f", "g"], dtype=h5py.string_dtype())
    cat = open_folder(str(tmp_path))
    s = [s for s in cat.streams() if s.sub == "modules/intent/left"][0]
    lab = s.info["col_labels"]
    assert s.info["names_source"] == "logged"
    assert lab["phase_posterior_1"] == "phase b" and lab["target_posterior_3"] == "target goal corridor"

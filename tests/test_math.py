import numpy as np
import pytest

from teleop_analysis import math_ops


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_se3_column_major():
    yaw, pitch, roll = 0.7, -0.3, 0.2
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.1, 0.2, 0.3]
    M = np.tile(T.flatten(order="F"), (5, 1))
    out = math_ops.se3_decompose(M)
    assert out["x [m]"][0] == pytest.approx(0.1)
    assert out["z [m]"][0] == pytest.approx(0.3)
    assert out["yaw [deg]"][0] == pytest.approx(np.degrees(yaw))
    assert out["pitch [deg]"][0] == pytest.approx(np.degrees(pitch))
    assert out["roll [deg]"][0] == pytest.approx(np.degrees(roll))
    q = np.array([out[k][0] for k in ("qw", "qx", "qy", "qz")])
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_quat_roundtrip():
    t = np.linspace(0, 1, 50)
    yaw = np.linspace(0, 3.5, 50)
    q = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)
    _, out = math_ops.evaluate("quat2rpy", [t] * 4, list(q.T), list("wxyz"), {})
    assert np.allclose(out["yaw [deg]"], np.degrees(yaw), atol=1e-6)


def test_norm_resamples_and_keeps_gaps():
    t1 = np.arange(0, 1, 0.01)
    t2 = np.arange(0, 1, 0.003)
    t2 = t2[(t2 < 0.4) | (t2 > 0.8)]
    _, out = math_ops.evaluate("norm", [t1, t2], [np.full_like(t1, 3.0), np.full_like(t2, 4.0)], ["a", "b"], {})
    n = out["norm"]
    assert n[10] == pytest.approx(5.0)
    assert np.isnan(n[60])


def test_derivative_and_expr():
    t = np.linspace(0, 1, 1001)
    _, d = math_ops.evaluate("derivative", [t], [t ** 2], ["y"], {})
    assert d["d/dt y"][500] == pytest.approx(1.0, abs=1e-3)
    _, e = math_ops.evaluate("expr", [t, t], [t, 2 * t], ["a", "b"], {"expression": "hypot(a, b)"})
    assert e["expr"][1000] == pytest.approx(np.sqrt(5))


def test_delay():
    t = np.arange(0, 5, 0.001)
    rng = np.random.default_rng(0)
    y = np.convolve(rng.standard_normal(len(t)), np.ones(50) / 50, mode="same")
    lag, corr = math_ops.estimate_delay(t, y, t + 0.0375, y)
    assert lag == pytest.approx(0.0375, abs=0.001)
    assert corr > 0.9

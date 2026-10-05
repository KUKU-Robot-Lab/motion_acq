"""Glove -> RH56F1 joints learned from example poses (10.05 calibration method).

The operator holds a set of example poses; each has a robot pose the hand should take
(configs/hands/nova2_to_rh56f1.yaml examples). The glove signals are coupled (bending
the ring finger moves the glove's thumb rotation reading, curling a finger brings its
tip to a still thumb), so a per-signal open/closed calibration moves the wrong robot
joints. Examples such as "index curled, thumb still" teach the map to keep them apart.

Model, per robot joint, on standardized inputs z = (x - offset - mean) / std:
    q = W z + b + sum_k alpha_k exp(-|z - c_k|^2 / (2 sigma^2))
an affine ridge fit over all example samples (smooth, never wild between poses) plus a
Gaussian residual centred on each example pose (reproduces the examples, including the
non-linear pinches). Pure numpy, Python 3.10 (the ROS hand node imports it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

RIDGE_LINEAR = 1e-2
RIDGE_RBF = 1e-3
MIN_STD = {"angle": 0.02, "tipdist": 2.0, "position": 2.0}  # floors for the input spread (rad, mm, mm)


class ExampleMapError(ValueError):
    pass


def input_kind(name: str) -> str:
    """angle (rad), tipdist (thumb-to-finger distance, mm) or position (thumb tip coordinate, mm)."""
    if name.startswith("tipdist_"):
        return "tipdist"
    return "position" if name.startswith("thumb_tip_") else "angle"


@dataclass(frozen=True)
class ExampleMap:
    inputs: tuple[str, ...]
    joints: tuple[str, ...]
    mean: np.ndarray  # (n_in,)
    std: np.ndarray  # (n_in,)
    weights: np.ndarray  # (n_joints, n_in)
    bias: np.ndarray  # (n_joints,)
    centers: np.ndarray  # (k, n_in) standardized
    alpha: np.ndarray  # (k, n_joints)
    sigma: float
    fallback: np.ndarray  # (n_in,) raw input used when one is missing (the open pose)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(0))  # raw input shift (quick re-zero)

    def _z(self, x: np.ndarray) -> np.ndarray:
        shift = self.offset if self.offset.size == x.size else 0.0
        return (x - shift - self.mean) / self.std

    def vector(self, signals: Mapping[str, float]) -> tuple[np.ndarray, list[str]]:
        """Raw input vector from the glove signals; missing ones (no tip data) take the open pose."""
        missing = [n for n in self.inputs if n not in signals]
        x = np.array([float(signals[n]) if n in signals else float(self.fallback[i])
                      for i, n in enumerate(self.inputs)])
        return x, missing

    def predict(self, signals: Mapping[str, float]) -> dict[str, float]:
        x, _ = self.vector(signals)
        z = self._z(x)
        q = self.weights @ z + self.bias
        if len(self.centers):
            d2 = np.sum((self.centers - z) ** 2, axis=1)
            q = q + np.exp(-d2 / (2.0 * self.sigma ** 2)) @ self.alpha
        return {j: float(v) for j, v in zip(self.joints, q)}

    def with_offset(self, offset: Sequence[float]) -> ExampleMap:
        return ExampleMap(self.inputs, self.joints, self.mean, self.std, self.weights, self.bias, self.centers,
                          self.alpha, self.sigma, self.fallback, np.asarray(offset, float))

    def to_dict(self) -> dict:
        return {
            "inputs": list(self.inputs), "joints": list(self.joints),
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "weights": self.weights.tolist(), "bias": self.bias.tolist(),
            "centers": self.centers.tolist(), "alpha": self.alpha.tolist(), "sigma": self.sigma,
            "fallback": self.fallback.tolist(), "offset": self.offset.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> ExampleMap:
        try:
            inputs, joints = tuple(raw["inputs"]), tuple(raw["joints"])
            arr = {k: np.asarray(raw[k], float) for k in ("mean", "std", "weights", "bias", "centers", "alpha",
                                                          "fallback")}
            offset = np.asarray(raw.get("offset") or [0.0] * len(inputs), float)
            sigma = float(raw["sigma"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExampleMapError(f"bad example map: {exc}") from exc
        n_in, n_j = len(inputs), len(joints)
        shapes = {"mean": (n_in,), "std": (n_in,), "weights": (n_j, n_in), "bias": (n_j,), "fallback": (n_in,)}
        for k, shape in shapes.items():
            if arr[k].shape != shape:
                raise ExampleMapError(f"example map {k} has shape {arr[k].shape}, expected {shape}")
        k = arr["centers"].shape[0] if arr["centers"].size else 0
        if k and (arr["centers"].shape != (k, n_in) or arr["alpha"].shape != (k, n_j)):
            raise ExampleMapError("example map centers / alpha do not match")
        finite = all(np.all(np.isfinite(a)) for a in arr.values()) and math.isfinite(sigma) and sigma > 0
        if not finite or np.any(arr["std"] <= 0) or offset.shape != (n_in,):
            raise ExampleMapError("example map has non-finite values, a zero spread or a bad offset")
        return cls(inputs, joints, arr["mean"], arr["std"], arr["weights"], arr["bias"],
                   arr["centers"].reshape(k, n_in), arr["alpha"].reshape(k, n_j), sigma, arr["fallback"], offset)


@dataclass(frozen=True)
class Conflict:
    """Two example poses the glove cannot tell apart but that ask for different robot poses."""
    poses: tuple[str, str]
    distance: float  # standardized input distance
    target_gap: float  # largest joint difference (rad)
    group: str = ""


def fit(inputs: Sequence[str], joints: Sequence[str],
        samples: Mapping[str, Sequence[Mapping[str, float]]], targets: Mapping[str, Mapping[str, float]],
        fallback_pose: str, *, min_distance: float = 1.0, min_target_gap: float = 0.15) -> ExampleMap:
    """Fit the map from example samples {pose: [signals]} and robot targets {pose: {joint: rad}}.

    Raises ExampleMapError listing conflicting poses (too close on the glove, different on
    the robot) so the operator redoes just those."""
    inputs, joints = tuple(inputs), tuple(joints)
    poses = [p for p in targets if p in samples and samples[p]]
    missing = [p for p in targets if p not in poses]
    if missing:
        raise ExampleMapError(f"no samples for poses {missing}")
    for p in poses:
        lacking = sorted({n for s in samples[p] for n in inputs if n not in s})
        if lacking:
            raise ExampleMapError(f"pose {p}: no glove values for {lacking} (glove tip data missing?)")
    x = np.array([[float(s[n]) for n in inputs] for p in poses for s in samples[p]])
    y = np.array([[float(targets[p][j]) for j in joints] for p in poses for _ in samples[p]])
    medians = {p: np.median(np.array([[float(s[n]) for n in inputs] for s in samples[p]]), axis=0) for p in poses}
    mean = x.mean(axis=0)
    floor = np.array([MIN_STD[input_kind(n)] for n in inputs])
    std = np.maximum(x.std(axis=0), floor)
    z = (x - mean) / std
    centers = np.array([(medians[p] - mean) / std for p in poses])
    conflicts = []
    for i, a in enumerate(poses):
        for b in poses[i + 1:]:
            dist = float(np.linalg.norm(centers[poses.index(a)] - centers[poses.index(b)]))
            gap = max(abs(targets[a][j] - targets[b][j]) for j in joints)
            if dist < min_distance and gap > min_target_gap:
                conflicts.append(Conflict((a, b), dist, gap))
    if conflicts:
        raise ConflictError(conflicts)
    # affine ridge
    za = np.hstack([z, np.ones((len(z), 1))])
    reg = RIDGE_LINEAR * len(z) * np.eye(za.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(za.T @ za + reg, za.T @ y)
    weights, bias = coef[:-1].T, coef[-1]
    # Gaussian residual on the example centres
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    off_diag = dists[~np.eye(len(poses), dtype=bool)]
    sigma = float(0.5 * np.median(off_diag)) if off_diag.size else 1.0
    resid = np.array([targets[p][j] for p in poses for j in joints]).reshape(len(poses), len(joints)) \
        - (centers @ weights.T + bias)
    phi = np.exp(-dists ** 2 / (2.0 * sigma ** 2))
    alpha = np.linalg.solve(phi + RIDGE_RBF * np.eye(len(poses)), resid)
    return ExampleMap(inputs, joints, mean, std, weights, bias, centers, alpha, sigma, medians[fallback_pose],
                      np.zeros(len(inputs)))


class ConflictError(ExampleMapError):
    def __init__(self, conflicts: Sequence[Conflict]) -> None:
        self.conflicts = list(conflicts)
        super().__init__("; ".join(f"{c.group + ': ' if c.group else ''}{c.poses[0]} / {c.poses[1]} look alike "
                                   f"on the glove (distance "
                                   f"{c.distance:.2f}) but differ on the robot by {c.target_gap:.2f} rad"
                                   for c in self.conflicts))


def fit_groups(groups: Mapping[str, tuple[Sequence[str], Sequence[str]]],
               samples: Mapping[str, Sequence[Mapping[str, float]]], targets: Mapping[str, Mapping[str, float]],
               fallback_pose: str) -> dict[str, ExampleMap]:
    """One map per joint group, each from its own inputs (10.05: with every input in one map,
    poses away from the examples fell back to the affine average and the thumb rotation sat
    at 1.2-1.7 rad). Conflicts of all groups are raised together."""
    models, conflicts = {}, []
    for name, (inputs, joints) in groups.items():
        try:
            models[name] = fit(inputs, joints, samples, targets, fallback_pose)
        except ConflictError as exc:
            conflicts += [Conflict(c.poses, c.distance, c.target_gap, name) for c in exc.conflicts]
    if conflicts:
        raise ConflictError(conflicts)
    return models

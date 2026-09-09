"""
map_matcher_v6.py
-----------------
Probabilistic Multi-Hypothesis Hidden Markov Model (HMM) Map Matching for 10 Hz Dead Reckoning.

Key components:
1. Multi-Candidate Tracking (top K=5 hypotheses).
2. Emission Probability:
   - Orthogonal distance to road centerline (Gaussian spatial decay)
   - Heading compatibility score: cos(vehicle_heading - road_heading)
3. Transition Probability:
   - Topological road connectivity (successors vs disconnected paths)
   - Path distance consistency vs integrated wheel/IMU displacement
   - Sharp U-turn & impossible maneuver penalty
4. Soft Snapping:
   - Avoids abrupt visual teleportation or trajectory jumping.
   - Gently regularizes drifting dead-reckoning trajectory towards the highest-likelihood road centerline.
"""

from __future__ import annotations

import math
import numpy as np


class RoadSegment:
    """Represents a directed road centerline segment between two waypoints."""

    def __init__(self, segment_id: int, p1: tuple[float, float], p2: tuple[float, float], successors: list[int] | None = None):
        self.segment_id = segment_id
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.vec = self.p2 - self.p1
        self.length = float(np.linalg.norm(self.vec)) + 1e-6
        self.unit_vec = self.vec / self.length
        self.heading = math.atan2(self.vec[1], self.vec[0])
        self.successors = successors or []

    def project_point(self, p: tuple[float, float]) -> tuple[float, np.ndarray, float]:
        """
        Projects point p onto this segment.
        Returns:
            dist: perpendicular distance in meters
            proj_pt: closest point coordinates on the segment
            s_frac: fractional distance along segment in [0, 1]
        """
        pt = np.array(p, dtype=np.float64)
        v = pt - self.p1
        s = float(np.dot(v, self.unit_vec))
        s_clamped = max(0.0, min(self.length, s))
        proj_pt = self.p1 + s_clamped * self.unit_vec
        dist = float(np.linalg.norm(pt - proj_pt))
        s_frac = s_clamped / self.length
        return dist, proj_pt, s_frac


class ProbabilisticHMMMapMatcherV6:
    def __init__(
        self,
        max_candidates: int = 5,
        search_radius_m: float = 35.0,
        sigma_dist: float = 8.0,
        sigma_heading: float = 0.5,
        soft_blend_alpha: float = 0.25,
    ):
        self.max_candidates = max_candidates
        self.search_radius = search_radius_m
        self.sigma_dist = sigma_dist
        self.sigma_heading = sigma_heading
        self.soft_blend_alpha = soft_blend_alpha
        self.segments: dict[int, RoadSegment] = {}
        self.active_hypotheses: list[dict] = []

    def build_road_graph_from_trajectory(self, lats: np.ndarray, lons: np.ndarray, segment_length_m: float = 25.0):
        """
        Constructs an offline topological road corridor graph from road survey data / coordinates.
        Converts lat/lon to local meters and discretizes into connected directed road segments.
        """
        self.segments.clear()
        self.active_hypotheses.clear()

        # Convert to local Cartesian coordinates (meters from first point)
        lat0, lon0 = lats[0], lons[0]
        pts = []
        for lat, lon in zip(lats, lons):
            dx = (lon - lon0) * 111139.0 * math.cos(math.radians(lat0))
            dy = (lat - lat0) * 111139.0
            pts.append((dx, dy))

        # Subsample into distinct road waypoints
        waypoints = [pts[0]]
        for pt in pts[1:]:
            d = math.hypot(pt[0] - waypoints[-1][0], pt[1] - waypoints[-1][1])
            if d >= segment_length_m:
                waypoints.append(pt)
        if len(waypoints) < len(pts) and math.hypot(pts[-1][0] - waypoints[-1][0], pts[-1][1] - waypoints[-1][1]) > 5.0:
            waypoints.append(pts[-1])

        # Create connected segments
        for i in range(len(waypoints) - 1):
            seg_id = i
            succ = [i + 1] if i + 1 < len(waypoints) - 1 else []
            seg = RoadSegment(seg_id, waypoints[i], waypoints[i + 1], successors=succ)
            self.segments[seg_id] = seg

    def match_step(
        self,
        current_x: float,
        current_y: float,
        current_heading: float,
        step_disp: float,
    ) -> tuple[float, float, float, dict]:
        """
        Executes one HMM Viterbi transition step at 10 Hz.
        Maintains multiple candidate hypotheses and gently guides the state towards the road.
        """
        if not self.segments:
            return current_x, current_y, current_heading, {"matched": False}

        pt = (current_x, current_y)

        # 1. Candidate Generation within search radius
        candidates = []
        for seg_id, seg in self.segments.items():
            dist, proj_pt, s_frac = seg.project_point(pt)
            if dist <= self.search_radius:
                # Heading compatibility
                head_diff = abs((current_heading - seg.heading + math.pi) % (2 * math.pi) - math.pi)
                # Emission log-likelihood: Gaussian on distance + heading
                log_emission = -(dist**2) / (2 * self.sigma_dist**2) - (head_diff**2) / (2 * self.sigma_heading**2)
                candidates.append({
                    "seg_id": seg_id,
                    "dist": dist,
                    "proj_pt": proj_pt,
                    "s_frac": s_frac,
                    "head_diff": head_diff,
                    "log_emission": log_emission,
                })

        if not candidates:
            # Out of search radius (e.g., unusual off-road maneuver) -> retain unconstrained dead reckoning
            return current_x, current_y, current_heading, {"matched": False}

        # Sort and prune to top candidates
        candidates.sort(key=lambda c: c["log_emission"], reverse=True)
        candidates = candidates[: self.max_candidates]

        # 2. Transition Scoring
        new_hypotheses = []
        for cand in candidates:
            if not self.active_hypotheses:
                # Initialization
                total_score = cand["log_emission"]
                new_hypotheses.append({**cand, "score": total_score})
            else:
                best_trans_score = -float("inf")
                for prev in self.active_hypotheses:
                    prev_seg = self.segments[prev["seg_id"]]
                    # Transition score based on topological connectivity
                    if cand["seg_id"] == prev["seg_id"]:
                        # Moving along same road
                        expected_ds = step_disp
                        actual_ds = (cand["s_frac"] - prev["s_frac"]) * prev_seg.length
                        ds_diff = abs(actual_ds - expected_ds)
                        log_trans = - (ds_diff**2) / (2 * 4.0**2)
                    elif cand["seg_id"] in prev_seg.successors:
                        # Connected next road
                        log_trans = -0.5
                    else:
                        # Disconnected / jump penalty
                        log_trans = -6.0

                    trans_score = prev["score"] + log_trans + cand["log_emission"]
                    if trans_score > best_trans_score:
                        best_trans_score = trans_score

                new_hypotheses.append({**cand, "score": best_trans_score})

        # Select highest-scoring hypothesis
        new_hypotheses.sort(key=lambda h: h["score"], reverse=True)
        self.active_hypotheses = new_hypotheses[: self.max_candidates]
        best_match = self.active_hypotheses[0]

        # 3. Soft Snapping: Blend dead-reckoning position towards road centerline
        proj_x, proj_y = best_match["proj_pt"]
        seg_heading = self.segments[best_match["seg_id"]].heading

        # Apply soft correction (e.g., 20-30% blend per step) to avoid trajectory jerk
        x_corr = (1.0 - self.soft_blend_alpha) * current_x + self.soft_blend_alpha * proj_x
        y_corr = (1.0 - self.soft_blend_alpha) * current_y + self.soft_blend_alpha * proj_y

        # Gentle heading alignment if within reasonable alignment
        if best_match["head_diff"] < 0.4:  # within ~23 degrees
            heading_corr = (1.0 - 0.15) * current_heading + 0.15 * seg_heading
        else:
            heading_corr = current_heading

        diag = {
            "matched": True,
            "seg_id": best_match["seg_id"],
            "dist_to_road": best_match["dist"],
            "score": best_match["score"],
        }
        return x_corr, y_corr, heading_corr, diag

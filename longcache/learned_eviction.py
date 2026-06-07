"""Phase 3: learned (contextual-bandit) KV eviction.

Each keep/evict decision is a contextual bandit. Context = per-token features available at
decision time: average attention received so far, position as a fraction of context, value
norm, and layer. Action = keep or evict. At a fixed budget the memory cost of "keep" is
constant, so the reward of keeping a token is the quality it preserves, proxied by the
attention it receives in the FUTURE. We collect rollouts (run the model, capture attention),
label each token with its future attention, train a linear reward model offline, and deploy it
as a cheap dot-product that scores eviction candidates.

H2O is the special case "score = raw past attention." So this directly tests whether learning
the importance function beats that heuristic. If it does not, the benchmark says so.

Inference is deliberately linear: scoring is (standardize -> dot product), a few mx ops per
eviction step, so the decode-time cost the verdict weighs stays small and measurable.
"""

import json
from dataclasses import asdict, dataclass
from typing import List

import mlx.core as mx
import numpy as np

from .heavy_hitter import TokenStreamRunner, capture_scores

FEATURE_NAMES = ["avg_attention", "position_fraction", "value_norm", "layer_fraction"]


@dataclass
class LinearPolicy:
    mean: List[float]
    scale: List[float]
    coef: List[float]
    intercept: float
    n_heads: int
    num_layers: int
    age: int
    future: int
    train_r2: float
    baseline_r2: float
    rows: int

    def to_json(self, path):
        with open(str(path), "w") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(str(path)) as handle:
            return cls(**json.load(handle))

    def mx_params(self):
        mean = mx.array(self.mean, dtype=mx.float32)
        scale = mx.array(self.scale, dtype=mx.float32)
        coef = mx.array(self.coef, dtype=mx.float32)
        return mean, scale, coef

    def summary(self):
        return {
            "features": FEATURE_NAMES,
            "coef": self.coef,
            "train_r2": self.train_r2,
            "baseline_r2_past_attention_only": self.baseline_r2,
            "rows": self.rows,
        }


def _value_norms(values):
    squared = (values.astype(mx.float32) * values.astype(mx.float32)).sum(axis=(1, 3))
    return mx.sqrt(squared)[0]


class RolloutCache:
    def __init__(self, layer_id, collector):
        self.layer_id = layer_id
        self.collector = collector
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        self.collector.on_values(self.layer_id, _value_norms(values))
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def record_scores(self, column):
        self.collector.on_column(self.layer_id, column)


class RolloutCollector:
    def __init__(self, age, future, n_heads, num_layers):
        self.age = age
        self.future = future
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.cum = {}
        self.vnorm = {}
        self.decision = {}
        self.X = []
        self.y = []

    def on_values(self, layer, norms):
        arr = np.asarray(norms, dtype=np.float64)
        if layer not in self.cum:
            self.cum[layer] = np.zeros(0, dtype=np.float64)
            self.vnorm[layer] = np.zeros(0, dtype=np.float64)
            self.decision[layer] = {}
        self.cum[layer] = np.concatenate([self.cum[layer], np.zeros(len(arr))])
        self.vnorm[layer] = np.concatenate([self.vnorm[layer], arr])

    def on_column(self, layer, column):
        col = np.asarray(column, dtype=np.float64)
        cum = self.cum[layer]
        cum[: len(col)] += col
        t = len(cum) - 1

        p_decision = t - self.age
        if p_decision >= 0:
            self.decision[layer][p_decision] = cum[p_decision]

        p_future = t - self.age - self.future
        if p_future >= 1 and p_future in self.decision[layer]:
            past = self.decision[layer].pop(p_future)
            future = cum[p_future] - past
            contributing = self.age + 1
            decision_offset = p_future + contributing
            avg_attention = past / (contributing * self.n_heads)
            position_fraction = p_future / decision_offset
            value_norm = self.vnorm[layer][p_future]
            layer_fraction = layer / self.num_layers
            self.X.append([avg_attention, position_fraction, value_norm, layer_fraction])
            self.y.append(future / (self.future * self.n_heads))


def collect_rollouts(runtime, token_ids, age, future):
    model = runtime.model
    n_heads = runtime.dims.num_heads
    num_layers = runtime.dims.num_layers
    collector = RolloutCollector(age, future, n_heads, num_layers)
    cache = [RolloutCache(i, collector) for i in range(num_layers)]
    ids = mx.array(token_ids)
    with capture_scores(model):
        for i in range(len(token_ids)):
            logits = model(ids[i : i + 1][None], cache=cache)
            mx.eval(logits)
    return np.asarray(collector.X, dtype=np.float64), np.asarray(collector.y, dtype=np.float64)


def train_policy(features, targets, runtime, age, future):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if features.shape[0] < 32:
        raise ValueError(
            f"Only {features.shape[0]} rollout rows; increase rollout_context."
        )

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)

    model = Ridge(alpha=1.0).fit(scaled, targets)
    train_r2 = float(model.score(scaled, targets))

    baseline = Ridge(alpha=1.0).fit(scaled[:, :1], targets)
    baseline_r2 = float(baseline.score(scaled[:, :1], targets))

    return LinearPolicy(
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=model.coef_.tolist(),
        intercept=float(model.intercept_),
        n_heads=runtime.dims.num_heads,
        num_layers=runtime.dims.num_layers,
        age=age,
        future=future,
        train_r2=train_r2,
        baseline_r2=baseline_r2,
        rows=int(features.shape[0]),
    )


class LearnedCache:
    def __init__(self, policy, budget, sink, recent, layer_id):
        if budget < sink + recent:
            raise ValueError(f"budget {budget} must be >= sink {sink} + recent {recent}.")
        self.budget = budget
        self.sink = sink
        self.recent = recent
        self.layer_id = layer_id
        self.n_heads = policy.n_heads
        self.layer_fraction = layer_id / policy.num_layers
        self._mean, self._scale, self._coef = policy.mx_params()
        self._intercept = policy.intercept
        self.keys = None
        self.values = None
        self.scores = None
        self.value_norms = None
        self.positions = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        new = keys.shape[2]
        positions = mx.arange(self.offset, self.offset + new).astype(mx.float32)
        norms = _value_norms(values)
        if self.keys is None:
            self.keys = keys
            self.values = values
            self.scores = mx.zeros((new,), dtype=mx.float32)
            self.value_norms = norms
            self.positions = positions
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
            self.scores = mx.concatenate([self.scores, mx.zeros((new,), dtype=mx.float32)])
            self.value_norms = mx.concatenate([self.value_norms, norms])
            self.positions = mx.concatenate([self.positions, positions])
        self.offset += new
        self._evict()
        return self.keys, self.values

    def _predict(self, lo, hi):
        lifetimes = self.offset - self.positions[lo:hi]
        avg_attention = self.scores[lo:hi] / (lifetimes * self.n_heads)
        position_fraction = self.positions[lo:hi] / self.offset
        value_norm = self.value_norms[lo:hi]
        layer_col = avg_attention * 0.0 + self.layer_fraction
        feats = mx.concatenate(
            [
                avg_attention[:, None],
                position_fraction[:, None],
                value_norm[:, None],
                layer_col[:, None],
            ],
            axis=1,
        )
        scaled = (feats - self._mean) / self._scale
        return scaled @ self._coef + self._intercept

    def _evict(self):
        size = self.keys.shape[2]
        if size <= self.budget:
            return
        n_drop = size - self.budget
        mid_start = self.sink
        mid_end = size - self.recent
        predicted = self._predict(mid_start, mid_end)
        order = mx.argsort(predicted)
        keep_local = mx.sort(order[n_drop:])
        head = mx.arange(mid_start).astype(mx.int32)
        kept = (keep_local + mid_start).astype(mx.int32)
        tail = mx.arange(mid_end, size).astype(mx.int32)
        keep_idx = mx.concatenate([head, kept, tail])
        self.keys = mx.take(self.keys, keep_idx, axis=2)
        self.values = mx.take(self.values, keep_idx, axis=2)
        self.scores = mx.take(self.scores, keep_idx, axis=0)
        self.value_norms = mx.take(self.value_norms, keep_idx, axis=0)
        self.positions = mx.take(self.positions, keep_idx, axis=0)

    def record_scores(self, column):
        if self.scores is None or column.shape[0] != self.scores.shape[0]:
            return
        self.scores = self.scores + column

    @property
    def state(self):
        return self.keys, self.values


class LearnedRunner(TokenStreamRunner):
    def __init__(self, runtime, policy, budget, sink, recent):
        self.policy = policy
        self.budget = budget
        self.sink = sink
        self.recent = recent
        num_layers = runtime.dims.num_layers
        super().__init__(
            runtime,
            lambda: [
                LearnedCache(policy, budget, sink, recent, i) for i in range(num_layers)
            ],
        )

"""The need gate: a calibrated probability that a query's reply needs the user's stored memories.

A turn like "set a timer for ten minutes" needs nothing the user told MIST before; a turn like
"what did I say my sister does?" needs nothing else. The fetch can skip, cut or serve memory by
how likely the turn is to need it, so each query gets a probability from a classifier on frozen
sentence embeddings, fit to a labelled bank of example turns written by two separate authors.

Two classifier forms compete, both pure numpy with no randomness, so a fit is reproducible from
the bank alone:

- `logistic`: `p = sigmoid(w . x + b)`, minimising `0.5 * ||w||^2 + C * sum_i s_i * logloss_i`
  with balanced weights `s_i = N / (2 * N_label(i))` and an unpenalised intercept, by Newton's
  method from zero.
- `knn`: the nearest-neighbour margin (mean of the `n` highest cosines to need rows minus the
  same to no-need rows), mapped to a probability by Platt scaling fit on the bank's
  leave-one-out margins with Platt's smoothed targets and the same balanced weights.

The form is chosen by cross-validation grouped by writer, at the operating point the gate works
at: the share of no-need turns closed when the threshold keeps a recall target of need turns.
Two thresholds give three bands: closed, band (a cut-down block) and open.

The shipped model is a record keyed by the bank, the embedder's name and revision, the query
prefix and the normalisation, so a change to any of them refuses to start instead of scoring
queries with a classifier fit to other vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from backend.errors import MemoryConfigError
from backend.interfaces import EmbeddingProvider

_HERE = Path(__file__).resolve().parent

DEFAULT_BANK_PATHS: tuple[Path, ...] = (
    _HERE / "need_bank" / "writer_a.jsonl",
    _HERE / "need_bank" / "writer_b.jsonl",
)
DEFAULT_MODEL_PATH: Path = _HERE / "need_model.json"

NORMALISATION = "l2"
LABELS = ("need", "no_need")

# Newton's method: a fixed cap and tolerance, so a fit either reaches the same answer every
# time or fails the same way. The cap is read at call time.
NEWTON_MAX_ITERATIONS = 100
NEWTON_TOLERANCE = 1e-8
_MAX_HALVINGS = 60
# Below this expected decrease (relative to the objective) a Newton step's change to the
# objective is lost in rounding, so the line search could not tell it from no step; the full
# step is taken instead, as it is in the region where Newton's method converges quadratically.
_ROUNDING_FLOOR = 1e-10

NEAREST_IDS = 3

_WRITER_FILE_RE = re.compile(r"^writer_([a-z0-9]+)\.jsonl$")
_ROW_FIELDS = frozenset({"id", "writer", "label", "category", "hard", "text"})


# --- the bank ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeedExemplar:
    """One labelled user turn of the bank.

    Attributes:
        id: Unique across every bank file, `<writer>-need-NNN` or `<writer>-no-need-NNN`.
        writer: The author's letter; cross-validation folds by it, so no author's phrasing is
            both fit and scored.
        label: `need` when the good answer depends on something the user told MIST before or
            keeps in notes, `no_need` when it depends only on the world, the device or the
            conversation in front of it.
        category: The label's category the turn was written for.
        hard: True for a turn worded like the other label's, kept apart in the hard closure.
        text: The turn itself.
    """

    id: str
    writer: str
    label: Literal["need", "no_need"]
    category: str
    hard: bool
    text: str


@dataclass(frozen=True, slots=True)
class NeedBank:
    """The exemplars and the hash of the bytes they were parsed from.

    Attributes:
        exemplars: Every row of every file, in file order then line order.
        sha256: `bank_sha256` of the same bytes the rows came from.
    """

    exemplars: tuple[NeedExemplar, ...]
    sha256: str


def bank_sha256(contents: Sequence[bytes]) -> str:
    """The bank's hash: the sha256 of each file's own sha256 digest, concatenated in order.

    Hashing each file first keeps the hash defined by file boundaries: moving a row from one
    writer's file to the other changes it.

    Args:
        contents: Each file's bytes, in writer order.

    Returns:
        A 64-character lower-case hex digest.
    """
    joined = b"".join(hashlib.sha256(data).digest() for data in contents)
    return hashlib.sha256(joined).hexdigest()


def _file_writer(path: Path) -> str:
    match = _WRITER_FILE_RE.match(path.name)
    if match is None:
        raise MemoryConfigError(f"{path}: a bank file is named writer_<writer>.jsonl")
    return match.group(1)


def _parse_row(path: Path, line_no: int, line: str, writer: str) -> NeedExemplar:
    where = f"{path}:{line_no}"
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise MemoryConfigError(f"{where}: malformed row ({exc.msg})") from exc
    if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
        raise MemoryConfigError(
            f"{where}: malformed row (an object with exactly {sorted(_ROW_FIELDS)})"
        )
    for name in ("id", "writer", "label", "category", "text"):
        if not isinstance(row[name], str):
            raise MemoryConfigError(f"{where}: malformed row ({name} is not a string)")
    if not isinstance(row["hard"], bool):
        raise MemoryConfigError(f"{where}: malformed row (hard is not a boolean)")
    if not row["id"] or not row["category"]:
        raise MemoryConfigError(f"{where}: malformed row (empty id or category)")
    if row["label"] not in LABELS:
        raise MemoryConfigError(f"{where}: unknown label {row['label']!r}")
    if not row["text"].strip():
        raise MemoryConfigError(f"{where}: empty text")
    if row["writer"] != writer:
        raise MemoryConfigError(
            f"{where}: writer {row['writer']!r} differs from the file's writer {writer!r}"
        )
    return NeedExemplar(
        row["id"], row["writer"], row["label"], row["category"], row["hard"], row["text"]
    )


def load_bank(paths: Sequence[Path] | None = None) -> NeedBank:
    """Read the bank files once and hash the bytes the rows were parsed from.

    Each file is read once and both the rows and the hash come from those bytes, so the hash
    always describes the rows the gate holds, even if a file changes on disk afterwards.

    Args:
        paths: The bank files, in writer order, each named `writer_<writer>.jsonl`;
            `DEFAULT_BANK_PATHS` when None.

    Returns:
        The exemplars and their hash.

    Raises:
        MemoryConfigError: A missing or unreadable file, a file not named for its writer or out
            of writer order, and, naming the file and line, a malformed row, an unknown label, a
            duplicate id, an empty text or a row of another writer; and a bank holding only one
            label.
    """
    paths = tuple(DEFAULT_BANK_PATHS if paths is None else paths)
    if not paths:
        raise MemoryConfigError("the need bank names no files")
    writers = [_file_writer(Path(path)) for path in paths]
    if writers != sorted(set(writers)):
        raise MemoryConfigError(
            f"the need bank files {[str(p) for p in paths]} are not one per writer in writer "
            "order"
        )
    contents: list[bytes] = []
    exemplars: list[NeedExemplar] = []
    seen: set[str] = set()
    for path, writer in zip(paths, writers):
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise MemoryConfigError(f"{path}: cannot read the need bank file ({exc})") from exc
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryConfigError(f"{path}: the need bank file is not UTF-8") from exc
        contents.append(data)
        for line_no, line in enumerate(text.splitlines(), start=1):
            exemplar = _parse_row(path, line_no, line, writer)
            if exemplar.id in seen:
                raise MemoryConfigError(f"{path}:{line_no}: duplicate id {exemplar.id!r}")
            seen.add(exemplar.id)
            exemplars.append(exemplar)
    if {e.label for e in exemplars} != set(LABELS):
        raise MemoryConfigError(
            f"the need bank {[str(p) for p in paths]} holds one label only; it needs both"
        )
    return NeedBank(tuple(exemplars), bank_sha256(contents))


# --- model records -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeedKey:
    """What a fitted gate depends on: change any of these and the fit no longer applies.

    Attributes:
        bank_sha256: The bank's `bank_sha256`.
        embedder_name: The gate embedder's name.
        embedder_revision: The gate embedder's revision.
        query_prefix: The text put before every bank row and query before embedding.
        normalisation: How vectors are scaled; `l2`, the only value, recorded so a later one
            changes the key.
    """

    bank_sha256: str
    embedder_name: str
    embedder_revision: str
    query_prefix: str
    normalisation: str = NORMALISATION


def need_model_key(key: NeedKey) -> str:
    """The hex sha256 of the key's five fields as canonical JSON (sorted keys, no spaces)."""
    payload = {f.name: getattr(key, f.name) for f in fields(NeedKey)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _differing_fields(wanted: NeedKey, have: NeedKey) -> list[str]:
    return [
        f"{f.name} {getattr(wanted, f.name)!r} (the record has {getattr(have, f.name)!r})"
        for f in fields(NeedKey)
        if getattr(wanted, f.name) != getattr(have, f.name)
    ]


@dataclass(frozen=True, slots=True)
class NeedModel:
    """One fitted gate: its key, its form and parameters, and its thresholds.

    Attributes:
        key: `need_model_key(self.parts())`.
        bank_sha256: As `NeedKey`.
        embedder_name: As `NeedKey`.
        embedder_revision: As `NeedKey`.
        query_prefix: As `NeedKey`.
        normalisation: As `NeedKey`.
        form: `logistic` or `knn`.
        params: `{"c", "w", "b"}` for `logistic`, `{"n", "a", "c"}` for `knn`.
        theta_low: Below it the gate is closed.
        theta_high: At or above it the gate is open; between the two it is band.
        band_k: How many block entries a band turn keeps.
        recall_target: The need recall the thresholds were set for.
        source: Which run fit it, for example `H18 2026-11-15`.
    """

    key: str
    bank_sha256: str
    embedder_name: str
    embedder_revision: str
    query_prefix: str
    normalisation: str
    form: Literal["logistic", "knn"]
    params: Mapping[str, object]
    theta_low: float
    theta_high: float
    band_k: int
    recall_target: float
    source: str

    def parts(self) -> NeedKey:
        """The record's five key fields."""
        return NeedKey(
            self.bank_sha256,
            self.embedder_name,
            self.embedder_revision,
            self.query_prefix,
            self.normalisation,
        )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite(where: str, name: str, value: object) -> float:
    if not _is_number(value):
        raise MemoryConfigError(f"{where}: {name} is not a number")
    if not math.isfinite(value):
        raise MemoryConfigError(f"{where}: {name} is not finite")
    return float(value)


def _string(where: str, name: str, value: object) -> str:
    if not isinstance(value, str):
        raise MemoryConfigError(f"{where}: {name} is not a string")
    return value


def _parse_params(where: str, form: str, params: object) -> Mapping[str, object]:
    if form == "logistic":
        if not isinstance(params, dict) or set(params) != {"c", "w", "b"}:
            raise MemoryConfigError(f"{where}: logistic params are an object of c, w and b")
        c = _finite(where, "params.c", params["c"])
        if c <= 0:
            raise MemoryConfigError(f"{where}: params.c is not positive")
        w = params["w"]
        if not isinstance(w, list) or not w:
            raise MemoryConfigError(f"{where}: params.w is not a non-empty list")
        weights = tuple(_finite(where, "params.w", v) for v in w)
        return MappingProxyType(
            {"c": c, "w": weights, "b": _finite(where, "params.b", params["b"])}
        )
    if not isinstance(params, dict) or set(params) != {"n", "a", "c"}:
        raise MemoryConfigError(f"{where}: knn params are an object of n, a and c")
    n = params["n"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise MemoryConfigError(f"{where}: params.n is not a positive integer")
    return MappingProxyType(
        {
            "n": n,
            "a": _finite(where, "params.a", params["a"]),
            "c": _finite(where, "params.c", params["c"]),
        }
    )


def _parse_record(where: str, record: object) -> NeedModel:
    expected = {
        "key",
        "bank_sha256",
        "embedder",
        "query_prefix",
        "normalisation",
        "form",
        "params",
        "theta_low",
        "theta_high",
        "band_k",
        "recall_target",
        "source",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise MemoryConfigError(f"{where}: a record is an object of {sorted(expected)}")
    embedder = record["embedder"]
    if not isinstance(embedder, dict) or set(embedder) != {"name", "revision"}:
        raise MemoryConfigError(f"{where}: embedder is an object of name and revision")
    form = record["form"]
    if form not in ("logistic", "knn"):
        raise MemoryConfigError(f"{where}: form {form!r} is not logistic or knn")
    normalisation = _string(where, "normalisation", record["normalisation"])
    if normalisation != NORMALISATION:
        raise MemoryConfigError(f"{where}: normalisation {normalisation!r} is not 'l2'")
    theta_low = _finite(where, "theta_low", record["theta_low"])
    theta_high = _finite(where, "theta_high", record["theta_high"])
    if theta_low > theta_high:
        raise MemoryConfigError(f"{where}: theta_low {theta_low} is above theta_high {theta_high}")
    band_k = record["band_k"]
    if isinstance(band_k, bool) or band_k not in (1, 2):
        raise MemoryConfigError(f"{where}: band_k {band_k!r} is not 1 or 2")
    model = NeedModel(
        key=_string(where, "key", record["key"]),
        bank_sha256=_string(where, "bank_sha256", record["bank_sha256"]),
        embedder_name=_string(where, "embedder.name", embedder["name"]),
        embedder_revision=_string(where, "embedder.revision", embedder["revision"]),
        query_prefix=_string(where, "query_prefix", record["query_prefix"]),
        normalisation=normalisation,
        form=form,
        params=_parse_params(where, form, record["params"]),
        theta_low=theta_low,
        theta_high=theta_high,
        band_k=band_k,
        recall_target=_finite(where, "recall_target", record["recall_target"]),
        source=_string(where, "source", record["source"]),
    )
    if model.key != need_model_key(model.parts()):
        raise MemoryConfigError(f"{where}: key {model.key!r} is not the hash of its key fields")
    return model


def load_models(path: Path | None = None) -> tuple[NeedModel, ...]:
    """Read the shipped gate models.

    Args:
        path: A file in need_model.json's format; `DEFAULT_MODEL_PATH` when None.

    Returns:
        The records, in file order.

    Raises:
        MemoryConfigError: A missing or malformed file, a non-finite number, `theta_low` above
            `theta_high`, `band_k` outside {1, 2}, two records with one key, or a key that is not
            the hash of its record's key fields.
    """
    path = DEFAULT_MODEL_PATH if path is None else Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MemoryConfigError(f"{path}: cannot read the need model file ({exc})") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MemoryConfigError(f"{path}: malformed need model file ({exc.msg})") from exc
    if not isinstance(document, dict) or set(document) != {"records"}:
        raise MemoryConfigError(f"{path}: the need model file is an object of records")
    records = document["records"]
    if not isinstance(records, list):
        raise MemoryConfigError(f"{path}: records is not a list")
    models: list[NeedModel] = []
    keys: set[str] = set()
    for index, record in enumerate(records):
        model = _parse_record(f"{path} record {index}", record)
        if model.key in keys:
            raise MemoryConfigError(f"{path}: two records with key {model.key}")
        keys.add(model.key)
        models.append(model)
    return tuple(models)


def select_model(models: Sequence[NeedModel], key: NeedKey) -> NeedModel:
    """The record fit for `key`.

    Raises:
        MemoryConfigError: No record has the key; the message names each field of `key` that
            differs from the record that differs in fewest fields (the first such in file
            order), so the operator sees what changed.
    """
    wanted = need_model_key(key)
    for model in models:
        if model.key == wanted:
            return model
    if not models:
        raise MemoryConfigError("the need model file holds no record")
    closest = min(models, key=lambda m: len(_differing_fields(key, m.parts())))
    differing = _differing_fields(key, closest.parts())
    raise MemoryConfigError(
        "no need model record has the configured key; the closest record differs in "
        + "; ".join(differing)
    )


# --- gate embedders ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeedEmbedderSpec:
    """How to load one gate embedder and what to put before each text.

    Attributes:
        query_prefix: The model card's query instruction, or empty.
        attn_implementation: Passed to the model load when set. ModernBERT models pick flash
            attention by themselves when the package is installed, which a CPU forward cannot
            run, so such a model names `sdpa` here.
    """

    query_prefix: str
    attn_implementation: str | None


NEED_EMBEDDERS: Mapping[str, NeedEmbedderSpec] = MappingProxyType(
    {
        "shared": NeedEmbedderSpec("", None),
        "bge-small-en-v1.5": NeedEmbedderSpec(
            "Represent this sentence for searching relevant passages: ", None
        ),
        "granite-embedding-small-english-r2": NeedEmbedderSpec("", "sdpa"),
    }
)


def need_embedder_kwargs(spec: NeedEmbedderSpec) -> dict[str, str]:
    """The keyword arguments a gate embedder's load takes beyond its name and device.

    Empty without an attention setting, so a caller passes the keyword only when a spec needs
    it, and a loader that does not take it is never handed it.
    """
    if spec.attn_implementation is None:
        return {}
    return {"attn_implementation": spec.attn_implementation}


def context_text(previous: str, query: str) -> str:
    """The text a follow-up is judged by: the previous user turn and the query, one per line."""
    return f"{previous}\n{query}"


# --- forms ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormSpec:
    """One classifier candidate.

    Attributes:
        form: `logistic` or `knn`.
        c: The logistic penalty's inverse strength; None for `knn`.
        n: The neighbours per label of the knn margin; None for `logistic`.
    """

    form: Literal["logistic", "knn"]
    c: float | None
    n: int | None

    def __post_init__(self) -> None:
        if self.form == "logistic":
            if self.n is not None or not _is_number(self.c) or not self.c > 0:
                raise ValueError(f"logistic takes a positive c and no n, not {self!r}")
            if not math.isfinite(self.c):
                raise ValueError(f"logistic takes a finite c, not {self!r}")
        elif self.form == "knn":
            if self.c is not None or not isinstance(self.n, int) or isinstance(self.n, bool):
                raise ValueError(f"knn takes an integer n and no c, not {self!r}")
            if self.n < 1:
                raise ValueError(f"knn takes n of at least 1, not {self!r}")
        else:
            raise ValueError(f"form {self.form!r} is not logistic or knn")

    def describe(self) -> str:
        """The form and its parameter, as error messages and reports name them."""
        if self.form == "logistic":
            return f"logistic c={self.c:g}"
        return f"knn n={self.n}"


FORM_CANDIDATES: tuple[FormSpec, ...] = (
    FormSpec("logistic", 0.1, None),
    FormSpec("logistic", 1.0, None),
    FormSpec("logistic", 10.0, None),
    FormSpec("logistic", 100.0, None),
    FormSpec("knn", None, 1),
    FormSpec("knn", None, 3),
    FormSpec("knn", None, 5),
)


class NeedFitError(ValueError):
    """A fit stopped before the gradient fell below the tolerance.

    A subclass of ValueError so cross-validation can mark only such a candidate failed, while a
    malformed input still raises.
    """


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # exp(-log(1 + exp(-z))), computed without overflow at either end.
    return np.exp(-np.logaddexp(0.0, -z))


def _unit_rows(vectors: object, what: str) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{what} are not a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} hold a value that is not finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{what} hold a vector of zero norm")
    return array / norms


def _need_mask(labels: Sequence[object], what: str = "labels") -> np.ndarray:
    mask = []
    for label in labels:
        if isinstance(label, bool | np.bool_):
            mask.append(bool(label))
        elif label == "need":
            mask.append(True)
        elif label == "no_need":
            mask.append(False)
        else:
            raise ValueError(f"{what} hold {label!r}, which is not need or no_need")
    return np.array(mask, dtype=bool)


def _balanced_weights(need: np.ndarray) -> np.ndarray:
    n_need = int(need.sum())
    n_no = need.size - n_need
    return np.where(need, need.size / (2 * n_need), need.size / (2 * n_no))


def _newton(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    penalised: np.ndarray,
    scale: float,
    describe: str,
) -> np.ndarray:
    """Minimise `0.5 * ||theta[penalised]||^2 + scale * sum_i weights_i * logloss_i`.

    `logloss_i` is the cross-entropy of `targets_i` against `sigmoid(features_i . theta)`.
    Newton's method from zero, with step halving, at most `NEWTON_MAX_ITERATIONS` steps,
    stopping when the gradient's largest component is below `NEWTON_TOLERANCE`.
    """
    max_iterations = NEWTON_MAX_ITERATIONS
    penalty = penalised.astype(np.float64)
    theta_ = np.zeros(features.shape[1])

    def objective(params: np.ndarray) -> float:
        z = features @ params
        loss = weights @ (np.logaddexp(0.0, z) - targets * z)
        return float(0.5 * (penalty * params) @ params + scale * loss)

    iterations = 0
    while True:
        z = features @ theta_
        p = _sigmoid(z)
        gradient = penalty * theta_ + scale * (features.T @ (weights * (p - targets)))
        largest = float(np.max(np.abs(gradient)))
        if largest < NEWTON_TOLERANCE:
            return theta_
        if iterations >= max_iterations:
            raise NeedFitError(
                f"{describe} reached {max_iterations} iterations before the gradient fell below "
                f"{NEWTON_TOLERANCE:g} (final largest gradient component {largest:.3g})"
            )
        curvature = scale * weights * p * _sigmoid(-z)
        hessian = np.diag(penalty) + features.T @ (curvature[:, None] * features)
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise NeedFitError(
                f"{describe} met a singular Hessian (final largest gradient component "
                f"{largest:.3g})"
            ) from exc
        current = objective(theta_)
        expected = -float(gradient @ direction)
        if expected <= _ROUNDING_FLOOR * max(1.0, abs(current)):
            theta_ = theta_ + direction
        else:
            step = 1.0
            for _ in range(_MAX_HALVINGS):
                candidate = theta_ + step * direction
                if objective(candidate) <= current:
                    theta_ = candidate
                    break
                step /= 2
            else:
                raise NeedFitError(
                    f"{describe} found no step that lowers the objective (final largest "
                    f"gradient component {largest:.3g})"
                )
        iterations += 1


def _knn_margins(
    queries: np.ndarray, bank: np.ndarray, need: np.ndarray, n: int, *, leave_one_out: bool
) -> np.ndarray:
    """Mean of the `n` highest cosines to need rows minus the same to no-need rows.

    With `leave_one_out`, `queries` is `bank` and each row's own cosine is excluded, so a row is
    never its own neighbour.
    """
    cosines = queries @ bank.T
    if leave_one_out:
        np.fill_diagonal(cosines, -np.inf)
    need_side = -np.sort(-cosines[:, need], axis=1)[:, :n]
    no_side = -np.sort(-cosines[:, ~need], axis=1)[:, :n]
    return need_side.mean(axis=1) - no_side.mean(axis=1)


def fit_form(vectors: object, labels: Sequence[object], spec: FormSpec) -> dict[str, object]:
    """Fit one candidate to labelled vectors; the result is the `params` of a model record.

    The vectors are scaled to unit length first, as the gate scales every vector.

    Args:
        vectors: One row per exemplar.
        labels: `need` or `no_need` (or True for need) per row.
        spec: The candidate.

    Returns:
        `{"c": C, "w": [...], "b": b}` for `logistic`; `{"n": n, "a": a, "c": c}` for `knn`,
        `p = sigmoid(a * margin + c)`. Plain floats, ready for JSON.

    Raises:
        NeedFitError: The fit reached the iteration cap first (a ValueError naming the form,
            its parameter and the final gradient).
        ValueError: Malformed input: a zero or non-finite vector, an unknown label, one label
            only, or a knn `n` above a label's row count less one.
    """
    x = _unit_rows(vectors, "vectors")
    need = _need_mask(labels)
    if need.size != x.shape[0]:
        raise ValueError(f"{need.size} labels for {x.shape[0]} vectors")
    if need.all() or not need.any():
        raise ValueError("fitting needs rows of both labels")
    weights = _balanced_weights(need)
    if spec.form == "logistic":
        features = np.hstack([x, np.ones((x.shape[0], 1))])
        penalised = np.ones(features.shape[1], dtype=bool)
        penalised[-1] = False
        params = _newton(
            features,
            need.astype(np.float64),
            weights,
            penalised=penalised,
            scale=float(spec.c),
            describe=spec.describe(),
        )
        return {"c": float(spec.c), "w": [float(v) for v in params[:-1]], "b": float(params[-1])}
    n = int(spec.n)
    fewest = min(int(need.sum()), int((~need).sum())) - 1
    if n > fewest:
        raise ValueError(
            f"{spec.describe()} needs at least {n + 1} rows of each label for leave-one-out "
            "margins"
        )
    margins = _knn_margins(x, x, need, n, leave_one_out=True)
    n_need, n_no = int(need.sum()), int((~need).sum())
    targets = np.where(need, (n_need + 1) / (n_need + 2), 1 / (n_no + 2))
    features = np.column_stack([margins, np.ones_like(margins)])
    params = _newton(
        features,
        targets,
        weights,
        penalised=np.zeros(2, dtype=bool),
        scale=1.0,
        describe=spec.describe(),
    )
    return {"n": n, "a": float(params[0]), "c": float(params[1])}


def _logits(
    x: np.ndarray,
    form: str,
    params: Mapping[str, object],
    bank: np.ndarray | None,
    bank_need: np.ndarray | None,
) -> np.ndarray:
    if form == "logistic":
        w = np.asarray(params["w"], dtype=np.float64)
        if w.shape != (x.shape[1],):
            raise ValueError(
                f"a weight vector of length {w.size} for vectors of length {x.shape[1]}"
            )
        return x @ w + float(params["b"])
    if form == "knn":
        if bank is None or bank_need is None:
            raise ValueError("knn scores against bank vectors and labels")
        if bank.shape[1] != x.shape[1]:
            raise ValueError(
                f"bank vectors of length {bank.shape[1]} for vectors of length {x.shape[1]}"
            )
        n = int(params["n"])
        if n > min(int(bank_need.sum()), int((~bank_need).sum())):
            raise ValueError(f"knn n={n} is above a label's row count in the bank")
        margins = _knn_margins(x, bank, bank_need, n, leave_one_out=False)
        return float(params["a"]) * margins + float(params["c"])
    raise ValueError(f"form {form!r} is not logistic or knn")


def probabilities(
    vectors: object,
    form: str,
    params: Mapping[str, object],
    *,
    bank_vectors: object | None = None,
    bank_labels: Sequence[object] | None = None,
) -> np.ndarray:
    """The need probability of each vector under a fitted form.

    Args:
        vectors: One row per query; scaled to unit length first.
        form: `logistic` or `knn`.
        params: As `fit_form` returns them or a model record holds them.
        bank_vectors: The rows the knn form was fit on (its neighbours); unused by `logistic`.
        bank_labels: Their labels.

    Returns:
        One probability per row.
    """
    x = _unit_rows(vectors, "vectors")
    bank = None if bank_vectors is None else _unit_rows(bank_vectors, "bank vectors")
    bank_need = None if bank_labels is None else _need_mask(bank_labels, "bank labels")
    if bank is not None and bank_need is not None and bank_need.size != bank.shape[0]:
        raise ValueError(f"{bank_need.size} bank labels for {bank.shape[0]} bank vectors")
    return _sigmoid(_logits(x, form, params, bank, bank_need))


def theta(probabilities: Sequence[float], r: float) -> float:
    """The threshold that keeps at least a share `r` of a set of need probabilities at or above it.

    Sort descending and take `k = ceil(r * N)`; the threshold is the midpoint of the k-th and the
    largest probability strictly below it, or half the k-th when none is below. So at least a
    share `r` has `p >= theta`, and no probability equals it: a probe sitting on the threshold
    would otherwise be decided by rounding.

    `r * N` is computed from `r` as written in decimal, so 0.88 of 25 is 22 even though the
    binary float of 0.88 is a hair above it.

    Raises:
        ValueError: An empty or non-finite set, `r` outside (0, 1], or a set where the rule's
            value equals an input (adjacent floats, or a k-th of zero).
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("theta needs a non-empty set of probabilities")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta's probabilities hold a value that is not finite")
    if not (0.0 < r <= 1.0):
        raise ValueError(f"theta's share {r} is outside (0, 1]")
    ordered = np.sort(values)[::-1]
    k = math.ceil(Fraction(str(r)) * values.size)
    kth = float(ordered[k - 1])
    below = ordered[ordered < kth]
    result = (kth + float(below[0])) / 2 if below.size else kth / 2
    if np.any(values == result):
        raise ValueError(f"theta({r}) cannot fall strictly between the set's values near {kth!r}")
    return result


# --- cross-validation ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CvResult:
    """One candidate's writer-grouped cross-validation.

    Fold 0 fits on the first writer (sorted) and scores the second; fold 1 the reverse.

    Attributes:
        spec: The candidate.
        closure: Per fold, per target: the share of the scored writer's no-need rows with
            `p < theta(R)`, `theta` over its need rows.
        hard_closure: The same over its hard no-need rows.
        auroc: Per fold, over the scored writer's rows.
        log_loss: Per fold, the mean cross-entropy.
        brier: Per fold, the mean squared error of `p`.
        failed: The fit reached the iteration cap in some fold; every tuple is then empty and
            `choose_form` skips the candidate.
    """

    spec: FormSpec
    closure: tuple[tuple[float, ...], ...]
    hard_closure: tuple[tuple[float, ...], ...]
    auroc: tuple[float, ...]
    log_loss: tuple[float, ...]
    brier: tuple[float, ...]
    failed: bool = False


def auroc(scores: Sequence[float], labels: Sequence[object]) -> float:
    """The probability a need row outscores a no-need row, ties counted half.

    Computed from average ranks (the Mann-Whitney statistic), which counts a tie half exactly.

    Args:
        scores: One score per row.
        labels: `need` or `no_need` (or a boolean, True for need) per row.

    Raises:
        ValueError: Rows of one label only, or a non-finite score.
    """
    values = np.asarray(scores, dtype=np.float64)
    need = _need_mask(labels)
    if values.shape != need.shape:
        raise ValueError(f"{need.size} labels for {values.size} scores")
    if not np.all(np.isfinite(values)):
        raise ValueError("auroc's scores hold a value that is not finite")
    positives = int(need.sum())
    negatives = need.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("auroc needs rows of both labels")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    rank_sum = float(ranks[need].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _share_below(values: np.ndarray, cut: float) -> float:
    return int(np.sum(values < cut)) / values.size


def grouped_cv(
    vectors: object,
    labels: Sequence[object],
    writers: Sequence[str],
    hard: Sequence[bool],
    specs: Sequence[FormSpec],
    *,
    targets: Sequence[float] = (0.95, 0.98),
) -> tuple[CvResult, ...]:
    """Cross-validate each candidate by writer: fit on one writer's rows, score the other's.

    Folding by writer keeps one author's phrasing out of both the fit and the scored rows, so the
    closure measures how the gate carries to turns worded by someone else. The closure is read
    at the operating point (`theta(R)` over the scored need rows) rather than over the whole
    curve, because only that point decides what the gate closes.

    Args:
        vectors: One row per exemplar.
        labels: `need` or `no_need` per row.
        writers: The writer per row; exactly two distinct writers.
        hard: Whether each row is a hard exemplar.
        specs: The candidates, in the order of the results.
        targets: The recall targets `R`.

    Returns:
        One result per candidate. A candidate whose fit reaches the iteration cap in either fold
        is marked failed.

    Raises:
        ValueError: Other than two writers, a writer without both labels or without a hard
            no-need row, mismatched lengths, or a target outside (0, 1].
    """
    x = np.asarray(vectors, dtype=np.float64)
    need = _need_mask(labels)
    writer_of = np.asarray(list(writers), dtype=object)
    hard_rows = np.asarray(list(hard), dtype=bool)
    if x.ndim != 2 or not (x.shape[0] == need.size == writer_of.size == hard_rows.size):
        raise ValueError("vectors, labels, writers and hard differ in length")
    for target in targets:
        if not (0.0 < target <= 1.0):
            raise ValueError(f"target {target} is outside (0, 1]")
    names = sorted(set(writer_of.tolist()))
    if len(names) != 2:
        raise ValueError(f"grouped_cv folds by two writers, not {names}")
    label_values = np.where(need, "need", "no_need")
    for name in names:
        rows = writer_of == name
        if need[rows].all() or not need[rows].any():
            raise ValueError(f"writer {name!r} lacks rows of both labels")
        if not np.any(rows & ~need & hard_rows):
            raise ValueError(f"writer {name!r} has no hard no-need row")
    folds = ((names[0], names[1]), (names[1], names[0]))
    results = []
    for spec in specs:
        closure, hard_closure, aurocs, log_losses, briers = [], [], [], [], []
        failed = False
        for fit_writer, scored_writer in folds:
            train = writer_of == fit_writer
            scored = writer_of == scored_writer
            try:
                params = fit_form(x[train], list(label_values[train]), spec)
            except NeedFitError:
                failed = True
                break
            p = np.asarray(
                probabilities(
                    x[scored],
                    spec.form,
                    params,
                    bank_vectors=x[train],
                    bank_labels=list(label_values[train]),
                ),
                dtype=np.float64,
            )
            y = need[scored]
            no_need_p = p[~y]
            hard_p = p[~y & hard_rows[scored]]
            cuts = [theta(p[y], target) for target in targets]
            closure.append(tuple(_share_below(no_need_p, cut) for cut in cuts))
            hard_closure.append(tuple(_share_below(hard_p, cut) for cut in cuts))
            aurocs.append(auroc(p, y))
            clipped = np.clip(p, 1e-15, 1 - 1e-15)
            log_losses.append(float(np.mean(-(y * np.log(clipped) + (~y) * np.log(1 - clipped)))))
            briers.append(float(np.mean((p - y) ** 2)))
        if failed:
            results.append(CvResult(spec, (), (), (), (), (), failed=True))
        else:
            results.append(
                CvResult(
                    spec,
                    tuple(closure),
                    tuple(hard_closure),
                    tuple(aurocs),
                    tuple(log_losses),
                    tuple(briers),
                )
            )
    return tuple(results)


def cv_score(result: CvResult) -> float:
    """The mean closure over the folds and the targets.

    Raises:
        ValueError: The candidate failed, so it has no closure.
    """
    if result.failed:
        raise ValueError(f"{result.spec.describe()} failed and has no score")
    values = [value for fold in result.closure for value in fold]
    if not values:
        raise ValueError(f"{result.spec.describe()} has no closure")
    return math.fsum(values) / len(values)


def _preference(spec: FormSpec) -> tuple[int, float, int]:
    # logistic before knn, then the smaller c, then the larger n: the smoother fit first.
    if spec.form == "logistic":
        return (0, float(spec.c), 0)
    return (1, 0.0, -int(spec.n))


def choose_form(results: Sequence[CvResult], *, no_need_rows: int) -> FormSpec:
    """The candidate cross-validation chooses: the smoothest within one standard error of best.

    The error is `sqrt(c * (1 - c) / m)`, `c` the best score and `m` the no-need rows in both
    files: the closure's binomial error alone. It ignores the variance of `theta(R)`, so it
    understates the true error and the rule prefers the smoother form less often than a full
    error would.

    Raises:
        ValueError: Every candidate failed, or `no_need_rows` is not positive.
    """
    if isinstance(no_need_rows, bool) or not isinstance(no_need_rows, int) or no_need_rows < 1:
        raise ValueError(f"no_need_rows {no_need_rows!r} is not a positive integer")
    scored = [(result.spec, cv_score(result)) for result in results if not result.failed]
    if not scored:
        raise ValueError("every candidate failed; no form can be chosen")
    best = max(score for _, score in scored)
    error = math.sqrt(best * (1 - best) / no_need_rows)
    eligible = [spec for spec, score in scored if score >= best - error]
    return min(eligible, key=_preference)


# --- the gate ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeedDecision:
    """The gate's decision for one query vector.

    Attributes:
        band: `closed` below `theta_low`, `band` from `theta_low` up to `theta_high`, `open` at
            `theta_high` and above.
        p: The need probability.
        theta_low: The model's lower threshold.
        theta_high: The model's upper threshold.
        band_k: The model's entry count for a band turn.
        need_ids: The nearest need exemplars by cosine, nearest first, for the trace.
        no_need_ids: The same for no-need exemplars.
    """

    band: Literal["closed", "band", "open"]
    p: float
    theta_low: float
    theta_high: float
    band_k: int
    need_ids: tuple[str, ...]
    no_need_ids: tuple[str, ...]


class NeedGate:
    """Scores query vectors against a fitted model and the bank it was fit on.

    The bank is embedded once, at construction; `decide` makes no model call, so the fetcher
    can run it on a vector it already has. `shares_query_vector` says the gate's embedder is the
    fetch's own model, so the fetcher passes its own query vector instead of embedding the query
    again; it is stated, not inferred from object identity, so a wrapped embedder does not embed
    twice.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        bank: NeedBank,
        model: NeedModel,
        *,
        embedder_name: str,
        embedder_revision: str,
        query_prefix: str,
        shares_query_vector: bool = False,
    ) -> None:
        if shares_query_vector and query_prefix:
            raise MemoryConfigError(
                "shares_query_vector reuses the fetch's own query vector, which has no prefix; "
                f"the gate's prefix is {query_prefix!r}"
            )
        key = NeedKey(bank.sha256, embedder_name, embedder_revision, query_prefix)
        if need_model_key(key) != model.key:
            differing = _differing_fields(key, model.parts())
            if not differing:
                raise MemoryConfigError("the need model's key is not the hash of its key fields")
            raise MemoryConfigError(
                "the need model was fit for another gate; it differs in " + "; ".join(differing)
            )
        if model.form not in ("logistic", "knn"):
            raise MemoryConfigError(f"need model form {model.form!r} is not logistic or knn")
        texts = [query_prefix + exemplar.text for exemplar in bank.exemplars]
        raw = embedder.generate_embeddings(texts)
        if len(raw) != len(texts):
            raise MemoryConfigError(
                f"the embedder returned {len(raw)} vectors for {len(texts)} rows"
            )
        if len({len(vector) for vector in raw}) != 1:
            raise MemoryConfigError("the embedder returned bank vectors of unequal length")
        try:
            vectors = _unit_rows(raw, "bank vectors")
        except ValueError as exc:
            raise MemoryConfigError(f"the need bank's embeddings are unusable: {exc}") from exc
        need = np.array([exemplar.label == "need" for exemplar in bank.exemplars], dtype=bool)
        dimension = vectors.shape[1]
        if model.form == "logistic":
            weights = model.params.get("w")
            if weights is None or len(weights) != dimension:
                length = None if weights is None else len(weights)
                raise MemoryConfigError(
                    f"the need model's weight vector has length {length}; the bank's vectors "
                    f"have length {dimension}"
                )
        else:
            n = model.params.get("n")
            fewest = min(int(need.sum()), int((~need).sum()))
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= fewest:
                raise MemoryConfigError(
                    f"the need model's knn n={n} is above a label's row count ({fewest})"
                )
        self._embedder = embedder
        self._model = model
        self._query_prefix = query_prefix
        self._shares_query_vector = shares_query_vector
        self._vectors = vectors
        self._need = need
        ids = np.array([exemplar.id for exemplar in bank.exemplars], dtype=object)
        self._need_ids = ids[need]
        self._no_need_ids = ids[~need]

    @property
    def embedder(self) -> EmbeddingProvider:
        """The gate's embedder."""
        return self._embedder

    @property
    def model(self) -> NeedModel:
        """The fitted model."""
        return self._model

    @property
    def query_prefix(self) -> str:
        """The text put before each query and bank row before embedding."""
        return self._query_prefix

    @property
    def shares_query_vector(self) -> bool:
        """Whether the fetch's own query vector is the gate's."""
        return self._shares_query_vector

    def embed_query(self, text: str) -> list[float]:
        """Embed a query as the gate's bank was embedded: with the prefix, in one call."""
        return self._embedder.generate_embedding(self._query_prefix + text)

    def decide(self, vector: Sequence[float]) -> NeedDecision:
        """The band, probability and nearest exemplars for one query vector.

        Raises:
            ValueError: A vector of another length than the bank's, a non-finite value or zero
                norm.
        """
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1 or array.size != self._vectors.shape[1]:
            raise ValueError(
                f"a query vector of length {array.size}; the bank's vectors have length "
                f"{self._vectors.shape[1]}"
            )
        x = _unit_rows(array[None, :], "the query vector")
        model = self._model
        p = float(_sigmoid(_logits(x, model.form, model.params, self._vectors, self._need))[0])
        if p < model.theta_low:
            band: Literal["closed", "band", "open"] = "closed"
        elif p < model.theta_high:
            band = "band"
        else:
            band = "open"
        cosines = self._vectors @ x[0]
        return NeedDecision(
            band=band,
            p=p,
            theta_low=model.theta_low,
            theta_high=model.theta_high,
            band_k=model.band_k,
            need_ids=_nearest(cosines[self._need], self._need_ids),
            no_need_ids=_nearest(cosines[~self._need], self._no_need_ids),
        )


def _nearest(cosines: np.ndarray, ids: np.ndarray) -> tuple[str, ...]:
    # Stable on the negated cosines: equal cosines keep bank order.
    order = np.argsort(-cosines, kind="mergesort")[:NEAREST_IDS]
    return tuple(str(ids[i]) for i in order)

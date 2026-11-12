"""The register of record: each labelling convention names a carrier a test can find.

ADR-054's v14 amendment (point 5) puts the register beside the conventions. A guide ends
in a `## Register` section whose table, `| Convention | Carrier | Class |`, holds a row
for each convention the guide states above it. A pack's guide is its `labelling.md`, and
its rows name rules of that pack.

A convention's label is derived from the lines above the register, never listed here:

- the N number of a line that opens a numbered rule, `**N17 ` or `- **N17 `;
- the text of a heading of the second level or deeper that has a line of its own before
  the next heading or numbered rule. A heading with no such line states nothing itself:
  the numbered rules under it do.

A row is a label's row when its key holds the label. The key is the Convention cell up to
its first colon, or the whole cell when it has none. An N number stands anywhere in the
key as a whole word; a heading opens it.

A Carrier cell names a carrier in one of these forms, and anything else names none:

    Rule 34 in `prompt-rules-11-to-37`    a prompt rule, and the pack's `prompt_fragment`
                                          rule whose text opens the rule of that number
    `team-fan-out`                        a rule of the pack, by its id
    Rule 5 of the engine core             a numbered rule of the prompt's `ENGINE_CORE`
    `backend.<module>.<function>`         an engine rule, by its function's qualified
                                          name, since the engine's rules declare no id

Several numbers may share one place: `Rules 13 and 16 in`, `Rules 8, 9 and 10 of`.

A pack's row names at least one rule of the pack. Its Class cell names the class of each
carrier the row names and no other: `prompt rule` for a prompt rule; `pre-LLM rule` for
any other rule that runs before the model; `payload rail`, `relation rail` or `id rail`
for a rule that runs after it, by the phase the rule declares; `engine rule` for an engine
function.

A row that names an `item_filter` rule quotes the rule's test in its Convention cell: the
words after `one of these ... words ...:`, up to the full stop. They are a list parted by
commas, whose last item may open with `and`, and `the phrase "x y"` quotes a phrase. The
quoted words are the rule's `keep_context_words`, in any order.
"""

from __future__ import annotations

import importlib
import inspect
import re
import shutil
import sys
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.knowledge.extraction import prompts
from backend.knowledge.ontologies.pack import PRE_LLM, ConventionRule, Pack, load_pack

BASE = "tests.unit.knowledge.ontologies.fixtures.base_pack"
PET = "tests.unit.knowledge.ontologies.fixtures.pet_pack"

REGISTER_HEADING = "## Register"
_COLUMNS = ("Convention", "Carrier", "Class")

PROMPT_RULE = "prompt rule"
PRE_LLM_RULE = "pre-LLM rule"
ENGINE_RULE = "engine rule"

_N_RULE_RE = re.compile(r"^(?:- )?\*\*(N\d+) ")
_N_LABEL_RE = re.compile(r"N\d+")
_HEADING_RE = re.compile(r"^#{2,6}\s+(.+?)\s*$")
_CELL_BREAK_RE = re.compile(r"(?<!\\)\|")
_TABLE_RULE_RE = re.compile(r":?-{3,}:?")
_CODE_RE = re.compile(r"`([^`]+)`")
_LIST_BREAK_RE = re.compile(r", | and ")
# A numbered rule as a prompt text opens it: `34. `, and a fixture's `F1. `.
_RULE_OPENING_RE = re.compile(r"^([A-Z]?\d+)\. ", re.MULTILINE)
_RULES_RE = re.compile(
    r"\bRules? (?P<numbers>[A-Z]?\d+(?:(?:, | and )[A-Z]?\d+)*) "
    r"(?:in `(?P<fragment>[^`]+)`|of the (?P<owner>engine core|active pack))"
)
_WORDS_RE = re.compile(r"\bone of these [^:.]*\bwords?\b[^:.]*: (?P<words>[^.]*)")
_PHRASE_RE = re.compile(r'the phrase "(.+)"')
_ENGINE_PREFIX = "backend."


# ---------------------------------------------------------------------------
# Reading a guide: its labels and its register
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One row of a register: its three cells as written."""

    convention: str
    carrier: str
    classes: str

    @property
    def key(self) -> str:
        """What the row is listed by: the Convention cell up to its first colon."""
        return self.convention.split(":", 1)[0]

    def holds(self, label: str) -> bool:
        """Whether this is a row of the convention `label`."""
        if _N_LABEL_RE.fullmatch(label):
            return re.search(rf"\b{label}\b", self.key) is not None
        return self.key.startswith(label)


@dataclass(frozen=True, slots=True)
class Register:
    """A guide as the check reads it.

    Attributes:
        labels: The label of each convention above the register, in the guide's order.
        rows: The register's rows.
        problems: What is wrong with the register's shape.
    """

    labels: tuple[str, ...]
    rows: tuple[Row, ...]
    problems: tuple[str, ...]


def convention_labels(lines: Sequence[str], not_conventions: Collection[str] = ()) -> list[str]:
    """The label of each convention `lines` state, by the rule of the module docstring.

    Args:
        lines: A guide's lines above its register.
        not_conventions: Headings that state no convention, for a guide that has
            sections about something else.
    """
    labels: list[str] = []
    waiting: str | None = None
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("#"):
            heading = _HEADING_RE.match(line)
            waiting = heading.group(1) if heading else None
            continue
        numbered = _N_RULE_RE.match(line)
        if numbered:
            labels.append(numbered.group(1))
        elif waiting is not None and waiting not in not_conventions:
            labels.append(waiting)
        waiting = None
    return labels


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in _CELL_BREAK_RE.split(line.strip().strip("|"))]


def _table(lines: Sequence[str]) -> tuple[list[Row], list[str]]:
    """The rows of the table in `lines`, and what is wrong with the table."""
    written = [line for line in lines if line.strip()]
    start = next((at for at, line in enumerate(written) if line.startswith("|")), None)
    if start is None:
        return [], ["the register holds no table"]
    head, rule, body = written[start], written[start + 1 : start + 2], written[start + 2 :]
    problems: list[str] = []
    if tuple(_cells(head)) != _COLUMNS:
        problems.append(f"the register's columns are not {', '.join(_COLUMNS)}: {head}")
    if not rule or not all(_TABLE_RULE_RE.fullmatch(cell) for cell in _cells(rule[0])):
        problems.append("the register's columns are not ruled off from its rows")
    rows: list[Row] = []
    for line in body:
        cells = _cells(line)
        if line.startswith("|") and len(cells) == len(_COLUMNS):
            rows.append(Row(*cells))
        else:
            problems.append(f"not a row of three cells: {line}")
    return rows, problems


def read_register(text: str, not_conventions: Collection[str] = ()) -> Register:
    """Read a guide: the labels above its `## Register` heading, and the table under it."""
    lines = text.splitlines()
    if REGISTER_HEADING not in lines:
        labels = convention_labels(lines, not_conventions)
        return Register(tuple(labels), (), (f"the guide has no `{REGISTER_HEADING}` section",))
    at = lines.index(REGISTER_HEADING)
    above, below = lines[:at], lines[at + 1 :]
    later = [line for line in below if line.startswith("#")]
    problems = [f"the register is not the guide's last section: {line}" for line in later]
    rows, table_problems = _table([line for line in below if not line.startswith("#")])
    return Register(
        tuple(convention_labels(above, not_conventions)),
        tuple(rows),
        tuple(problems + table_problems),
    )


def _missing_rows(register: Register) -> list[str]:
    return [
        f"{label}: no row of the register holds this label"
        for label in register.labels
        if not any(row.holds(label) for row in register.rows)
    ]


# ---------------------------------------------------------------------------
# What a Carrier cell names
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Carriers:
    """The carriers one Carrier cell names.

    Attributes:
        in_fragments: (rule number, id of the rule said to hold it) pairs.
        engine_core: Rule numbers said to be the engine core's.
        active_pack: Rule numbers said to be the active pack's.
        ids: Every name in backticks that is not an engine function's.
        functions: The qualified names of engine functions.
    """

    in_fragments: tuple[tuple[str, str], ...]
    engine_core: tuple[str, ...]
    active_pack: tuple[str, ...]
    ids: tuple[str, ...]
    functions: tuple[str, ...]


def carriers(cell: str) -> Carriers:
    """Read a Carrier cell by the forms of the module docstring."""
    in_fragments: list[tuple[str, str]] = []
    engine_core: list[str] = []
    active_pack: list[str] = []
    for match in _RULES_RE.finditer(cell):
        numbers = _LIST_BREAK_RE.split(match["numbers"])
        if match["fragment"]:
            in_fragments.extend((number, match["fragment"]) for number in numbers)
        elif match["owner"] == "engine core":
            engine_core.extend(numbers)
        else:
            active_pack.extend(numbers)
    names = _CODE_RE.findall(cell)
    return Carriers(
        in_fragments=tuple(in_fragments),
        engine_core=tuple(engine_core),
        active_pack=tuple(active_pack),
        ids=tuple(name for name in names if not name.startswith(_ENGINE_PREFIX)),
        functions=tuple(name for name in names if name.startswith(_ENGINE_PREFIX)),
    )


def quoted_words(convention: str) -> tuple[str, ...]:
    """The words of the `item_filter` test a Convention cell quotes; none when it quotes none."""
    match = _WORDS_RE.search(convention)
    if match is None:
        return ()
    words: list[str] = []
    for item in match["words"].split(","):
        item = item.strip().removeprefix("and ").strip()
        phrase = _PHRASE_RE.fullmatch(item)
        words.append(phrase.group(1) if phrase else item)
    return tuple(word for word in words if word)


def engine_core_rules() -> frozenset[str]:
    """The number of each rule the engine's prompt core states."""
    core = "".join(part for part in prompts.ENGINE_CORE if isinstance(part, str))
    return frozenset(_RULE_OPENING_RE.findall(core))


def engine_function(name: str) -> object | None:
    """The function a qualified name names, a module's or a class's; None when there is none."""
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            found: object = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attribute in parts[cut:]:
            found = getattr(found, attribute, None)
        return found if inspect.isfunction(found) else None
    return None


def _engine_carriers(named: Carriers, where: str) -> tuple[set[str], list[str]]:
    """The classes of the engine carriers a cell names, and the ones that do not exist."""
    classes: set[str] = set()
    problems: list[str] = []
    core = engine_core_rules()
    for number in named.engine_core:
        if number in core:
            classes.add(PROMPT_RULE)
        else:
            problems.append(f"{where}: the engine core holds no Rule {number}")
    for name in named.functions:
        if engine_function(name) is None:
            problems.append(f"{where}: `{name}` is no function of the engine")
        else:
            classes.add(ENGINE_RULE)
    return classes, problems


def _class_problems(row: Row, where: str, carried_by: set[str]) -> list[str]:
    if set(_LIST_BREAK_RE.split(row.classes)) == carried_by:
        return []
    classes = " and ".join(sorted(carried_by))
    return [f"{where}: its Class cell reads `{row.classes}`, and its carriers are {classes}"]


# ---------------------------------------------------------------------------
# A pack's register
# ---------------------------------------------------------------------------


def _class_of(rule: ConventionRule) -> str:
    if rule.kind == "prompt_fragment":
        return PROMPT_RULE
    if rule.phase == PRE_LLM:
        return PRE_LLM_RULE
    return f"{rule.phase} rail"


def _pack_row_problems(row: Row, pack: Pack) -> list[str]:
    where = f"row `{row.key}`"
    rules = {rule.id: rule for rule in pack.rules}
    named = carriers(row.carrier)
    carried_by, problems = _engine_carriers(named, where)
    if not named.ids:
        problems.append(f"{where}: names no rule of the pack")
    for rule_id in named.ids:
        rule = rules.get(rule_id)
        if rule is None:
            problems.append(f"{where}: `{rule_id}` is no rule of the pack")
            continue
        carried_by.add(_class_of(rule))
        if rule.kind != "item_filter":
            continue
        quoted, kept = sorted(quoted_words(row.convention)), sorted(rule.params.keep_context_words)
        if quoted != kept:
            problems.append(
                f"{where}: quotes the words {', '.join(quoted) or 'none'}, "
                f"and `{rule_id}` keeps {', '.join(kept) or 'none'}"
            )
    for number, fragment in named.in_fragments:
        rule = rules.get(fragment)
        if rule is None:
            continue
        opened = (
            _RULE_OPENING_RE.findall(rule.params.text) if rule.kind == "prompt_fragment" else ()
        )
        if number not in opened:
            problems.append(f"{where}: `{fragment}` holds no Rule {number}")
    for number in named.active_pack:
        problems.append(f"{where}: Rule {number} is named with no fragment of the pack")
    return problems or _class_problems(row, where, carried_by)


def pack_register_problems(pack: Pack) -> list[str]:
    """What is wrong with the register of a pack's guide; empty when it holds.

    Args:
        pack: A loaded pack. Its guide is `pack.labelling`.
    """
    register = read_register(pack.labelling)
    problems = [*register.problems, *_missing_rows(register)]
    for row in register.rows:
        problems.extend(_pack_row_problems(row, pack))
    return problems


# ---------------------------------------------------------------------------
# The fixture packs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", [BASE, PET], ids=["base", "pet"])
def test_the_register_of_a_fixture_pack_holds(package: str):
    pack = load_pack(package)

    assert pack_register_problems(pack) == []


def test_the_fixture_packs_labels_are_derived_from_its_lines():
    register = read_register(load_pack(PET).labelling)

    # `### Crews` and `### Pets` have no line of their own, so their rules are the labels.
    assert register.labels == ("N1", "N2", "Unchecked items", "N18")


@pytest.mark.parametrize(
    ("lines", "labels"),
    [
        pytest.param(["**N4 A rule.** Its text."], ["N4"], id="a-numbered-paragraph"),
        pytest.param(["- **N12 A rule.** Its text."], ["N12"], id="a-numbered-bullet"),
        pytest.param(["### Moorings", "", "A line of its own."], ["Moorings"], id="a-heading"),
        pytest.param(["## Moorings", "A line of its own.", "Another."], ["Moorings"], id="once"),
        pytest.param(["### Moorings", "", "**N4 A rule.**"], ["N4"], id="a-heading-with-no-line"),
        pytest.param(["### Moorings", "### Berths", "Text."], ["Berths"], id="an-empty-heading"),
        pytest.param(["# The guide", "", "What the guide is for."], [], id="the-title"),
        pytest.param(["Text that cites **N4 ** inside a line."], [], id="a-number-inside-a-line"),
    ],
)
def test_a_label_is_a_rule_number_or_a_heading_with_a_line_of_its_own(lines, labels):
    assert convention_labels(lines) == labels


def test_a_heading_named_as_no_convention_gives_no_label():
    lines = ["## Moorings", "Text.", "## How the files are named", "Text."]

    assert convention_labels(lines, {"How the files are named"}) == ["Moorings"]


def test_a_carrier_cell_is_read_by_its_forms():
    cell = (
        "Rules 13 and 16 in `late-rules`, `berth-rail`, Rules 8, 9 and 10 of the engine core, "
        "Rule 23 of the active pack, and `backend.knowledge.extraction.id_rails.n1_rail`"
    )

    assert carriers(cell) == Carriers(
        in_fragments=(("13", "late-rules"), ("16", "late-rules")),
        engine_core=("8", "9", "10"),
        active_pack=("23",),
        ids=("late-rules", "berth-rail"),
        functions=("backend.knowledge.extraction.id_rails.n1_rail",),
    )


def test_a_rule_number_with_no_place_names_no_carrier():
    assert carriers("Rule 34's prompt fragment, and a rule that fans out") == Carriers(
        (), (), (), (), ()
    )


# ---------------------------------------------------------------------------
# The failing cases, each on a copy of the base fixture pack under `tmp_path`
# ---------------------------------------------------------------------------

Edit = Callable[[str], str]

_N2_ROW = "| N2 A skipper's rank | `skipper-outranks` | relation rail |"
_N1_CARRIER = "Rule F1 in `crew-rule-text`, and `crew-fan-out`"
_N1_CLASSES = "prompt rule and relation rail"


def _swap(old: str, new: str) -> Edit:
    """An edit that puts `new` where a file holds `old`, which it must hold once."""

    def edit(text: str) -> str:
        assert text.count(old) == 1, f"the copy does not hold this once: {old}"
        return text.replace(old, new)

    return edit


@pytest.fixture
def base_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Pack]]:
    """Load a copy of the base fixture pack from `tmp_path`, edited before it is loaded."""
    source = Path(str(importlib.import_module(BASE).__file__)).parent
    package = f"copy_{tmp_path.name}"
    monkeypatch.syspath_prepend(str(tmp_path))

    def load(*, guide: Edit | None = None, conventions: Edit | None = None) -> Pack:
        target = tmp_path / package
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        for name, edit in (("labelling.md", guide), ("conventions.py", conventions)):
            if edit is not None:
                path = target / name
                path.write_text(edit(path.read_text(encoding="utf-8")), encoding="utf-8")
        return load_pack(package)

    yield load

    for name in [name for name in sys.modules if name.split(".")[0] == package]:
        del sys.modules[name]


def test_an_unedited_copy_holds(base_copy):
    pack = base_copy()

    assert pack_register_problems(pack) == []


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        pytest.param(
            f"\n{REGISTER_HEADING}\n",
            f"\n**N3 Night watches.** A deckhand on the night watch keeps it.\n\n{REGISTER_HEADING}\n",
            "N3: no row of the register holds this label",
            id="a-numbered-rule-with-no-row",
        ),
        pytest.param(
            "| Unchecked items: a",
            "| Unticked items: a",
            "Unchecked items: no row of the register holds this label",
            id="a-heading-with-no-row",
        ),
        pytest.param(
            f"{_N2_ROW}\n",
            "",
            "N2: no row of the register holds this label",
            id="a-row-taken-out",
        ),
        pytest.param(
            _N2_ROW,
            "| N2 A skipper's rank | the rail that drops the edge | relation rail |",
            "row `N2 A skipper's rank`: names no rule of the pack",
            id="a-row-that-names-no-carrier",
        ),
        pytest.param(
            _N2_ROW,
            "| N2 A skipper's rank | | relation rail |",
            "row `N2 A skipper's rank`: names no rule of the pack",
            id="an-empty-carrier-cell",
        ),
        pytest.param(
            _N2_ROW,
            "| N2 A skipper's rank | Rule 9 of the engine core | prompt rule |",
            "row `N2 A skipper's rank`: names no rule of the pack",
            id="a-row-that-names-an-engine-rule-alone",
        ),
        pytest.param(
            "`skipper-outranks`",
            "`skipper-salutes`",
            "row `N2 A skipper's rank`: `skipper-salutes` is no rule of the pack",
            id="a-carrier-id-that-is-not-in-the-pack",
        ),
        pytest.param(
            "plural: drill, and",
            "plural: muster, and",
            "row `Unchecked items`: quotes the words muster, watch list, "
            "and `unchecked-items` keeps drill, watch list",
            id="a-quoted-word-that-differs",
        ),
        pytest.param(
            ', and the phrase "watch list" |',
            " |",
            "row `Unchecked items`: quotes the words drill, "
            "and `unchecked-items` keeps drill, watch list",
            id="a-quoted-phrase-left-out",
        ),
        pytest.param(
            "Rule F1 in",
            "Rule F9 in",
            "row `N1 Serving under the skipper`: `crew-rule-text` holds no Rule F9",
            id="a-prompt-rule-its-fragment-does-not-hold",
        ),
        pytest.param(
            _N1_CARRIER,
            "Rule F1 in `crew-fan-out`",
            "row `N1 Serving under the skipper`: `crew-fan-out` holds no Rule F1",
            id="a-prompt-rule-in-a-rule-that-is-no-fragment",
        ),
        pytest.param(
            _N1_CARRIER,
            f"{_N1_CARRIER}, and Rule 6 of the engine core",
            "row `N1 Serving under the skipper`: the engine core holds no Rule 6",
            id="an-engine-core-rule-that-does-not-exist",
        ),
        pytest.param(
            _N1_CARRIER,
            f"{_N1_CARRIER}, and `backend.knowledge.extraction.id_rails.n9_rail`",
            "row `N1 Serving under the skipper`: "
            "`backend.knowledge.extraction.id_rails.n9_rail` is no function of the engine",
            id="an-engine-function-that-does-not-exist",
        ),
        pytest.param(
            _N1_CARRIER,
            f"{_N1_CARRIER}, and `backend.knowledge.extraction.id_rails.N1_WORDS`",
            "row `N1 Serving under the skipper`: "
            "`backend.knowledge.extraction.id_rails.N1_WORDS` is no function of the engine",
            id="an-engine-name-that-is-no-function",
        ),
        pytest.param(
            _N1_CARRIER,
            f"{_N1_CARRIER}, and Rule F1 of the active pack",
            "row `N1 Serving under the skipper`: Rule F1 is named with no fragment of the pack",
            id="a-pack-rule-named-with-no-fragment",
        ),
        pytest.param(
            _N2_ROW,
            "| N2 A skipper's rank | `skipper-outranks` | id rail |",
            "row `N2 A skipper's rank`: its Class cell reads `id rail`, "
            "and its carriers are relation rail",
            id="a-class-that-is-not-the-carriers",
        ),
        pytest.param(
            f"{_N1_CARRIER} | {_N1_CLASSES} |",
            f"{_N1_CARRIER} | prompt rule |",
            "row `N1 Serving under the skipper`: its Class cell reads `prompt rule`, "
            "and its carriers are prompt rule and relation rail",
            id="a-class-left-out",
        ),
        pytest.param(
            "| Convention | Carrier | Class |",
            "| Convention | Carried by | Class |",
            "the register's columns are not Convention, Carrier, Class: "
            "| Convention | Carried by | Class |",
            id="other-columns",
        ),
        pytest.param(
            "|---|---|---|\n",
            "",
            "the register's columns are not ruled off from its rows",
            id="no-rule-under-the-columns",
        ),
        pytest.param(
            _N2_ROW,
            "| N2 A skipper's rank | `skipper-outranks` |",
            "not a row of three cells: | N2 A skipper's rank | `skipper-outranks` |",
            id="a-row-of-two-cells",
        ),
    ],
)
def test_a_register_that_does_not_hold_fails(base_copy, old: str, new: str, problem: str):
    pack = base_copy(guide=_swap(old, new))

    assert problem in pack_register_problems(pack)


def test_a_rule_whose_words_differ_from_the_quoted_test_fails(base_copy):
    pack = base_copy(conventions=_swap('("drill", "watch list")', '("drill", "muster roll")'))

    assert pack_register_problems(pack) == [
        "row `Unchecked items`: quotes the words drill, watch list, "
        "and `unchecked-items` keeps drill, muster roll"
    ]


def test_the_quoted_words_may_stand_in_another_order(base_copy):
    pack = base_copy(conventions=_swap('("drill", "watch list")', '("watch list", "drill")'))

    assert pack_register_problems(pack) == []


def test_a_guide_with_no_register_fails(base_copy):
    pack = base_copy(guide=lambda text: text.split(REGISTER_HEADING)[0])

    assert pack_register_problems(pack) == [
        "the guide has no `## Register` section",
        "N1: no row of the register holds this label",
        "N2: no row of the register holds this label",
        "Unchecked items: no row of the register holds this label",
    ]


def test_a_register_with_no_table_fails(base_copy):
    pack = base_copy(guide=lambda text: text.split(REGISTER_HEADING)[0] + REGISTER_HEADING + "\n")

    assert "the register holds no table" in pack_register_problems(pack)


def test_a_register_that_is_not_the_last_section_fails(base_copy):
    pack = base_copy(guide=lambda text: text + "\n## Afterword\n\nA later section.\n")

    assert pack_register_problems(pack) == [
        "the register is not the guide's last section: ## Afterword",
        "not a row of three cells: A later section.",
    ]


def test_an_engine_rule_beside_a_pack_rule_is_a_carrier_of_its_own_class(base_copy):
    carrier = f"{_N1_CARRIER}, and `backend.knowledge.extraction.id_rails.n1_rail`"
    pack = base_copy(
        guide=_swap(
            f"{_N1_CARRIER} | {_N1_CLASSES} |",
            f"{carrier} | {_N1_CLASSES} and engine rule |",
        )
    )

    assert pack_register_problems(pack) == []

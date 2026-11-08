def test_metric_id_is_value_then_unit_regardless_of_surface():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("12000", "requests-per-second") == "12000-requests-per-second"
    assert canonical_metric_id("87", "percent-coverage") == "87-percent-coverage"


def test_metric_id_slugifies_components():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("12,000", "Requests / Second") == "12000-requests-second"


def test_string_metric_id_moves_number_to_front():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id("requests-per-second-12000") == "12000-requests-per-second"


def test_string_metric_id_already_value_first_unchanged():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id("12000-requests-per-second") == "12000-requests-per-second"
    assert canonical_metric_id_from_id("87-percent") == "87-percent"


def test_string_metric_id_no_or_multiple_numbers_unchanged():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id("backend-work") == "backend-work"
    assert canonical_metric_id_from_id("3-of-5-passes") == "3-of-5-passes"  # 2 numbers -> unchanged


def test_metric_id_without_quantity_is_value_then_unit_byte_for_byte():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("120", "ms") == "120-ms"
    assert canonical_metric_id("120", "ms", None) == "120-ms"
    assert canonical_metric_id("120", "ms", "") == "120-ms"
    assert canonical_metric_id(87, "percent-coverage") == "87-percent-coverage"


def test_metric_id_joins_quantity_after_unit():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    # A p95 and a p99 latency of one value and unit no longer collide on one id.
    assert canonical_metric_id("120", "ms", "p95 latency") == "120-ms-p95-latency"
    assert canonical_metric_id("120", "ms", "p99 latency") == "120-ms-p99-latency"
    assert canonical_metric_id("87", "percent", "Coverage") == "87-percent-coverage"


def test_metric_id_drops_quantity_the_unit_already_ends_with():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("87", "percent-coverage", "coverage") == "87-percent-coverage"
    assert canonical_metric_id("120", "ms-p95-latency", "P95 Latency") == "120-ms-p95-latency"
    assert canonical_metric_id("3", "latency", "latency") == "3-latency"


def test_metric_id_keeps_quantity_that_only_shares_trailing_characters():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    # The tail check compares whole slug tokens: the unit `ms` does not end with the token `s`.
    assert canonical_metric_id("120", "ms", "s") == "120-ms-s"


def test_metric_id_without_unit_joins_value_and_quantity():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("180", None, "max queue depth") == "180-max-queue-depth"
    assert canonical_metric_id("180", "", "max queue depth") == "180-max-queue-depth"

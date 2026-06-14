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

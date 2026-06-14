def test_metric_id_is_value_then_unit_regardless_of_surface():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("12000", "requests-per-second") == "12000-requests-per-second"
    assert canonical_metric_id("87", "percent-coverage") == "87-percent-coverage"


def test_metric_id_slugifies_components():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("12,000", "Requests / Second") == "12000-requests-second"

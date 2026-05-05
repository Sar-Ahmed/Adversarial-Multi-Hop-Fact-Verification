"""Phase 00 placeholder smoke test.

Replaced by tests/test_smoke.py in Phase 02 (5 fixed claims, end-to-end
through the real pipeline). This file exists so `make smoke` exits 0
on a clean Phase 00 checkout.
"""

import pytest


@pytest.mark.smoke
def test_imports_and_helpers() -> None:
    """Verify the Phase 00 helpers import and run without error."""
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging(json_sink=False)


@pytest.mark.smoke
def test_subclaim_field_name_regression() -> None:
    """Explicit regression test for the V2 schema bug.

    V2 crashed on every claim because its decomposer instantiated
    `SubClaim(sub_claim_id=...)` against a schema that defines the field as
    `id`. Phase 02 introduces the real `SubClaim`; this test ensures we never
    re-introduce that bug. For now it just records intent.
    """
    pytest.skip("Schema not yet defined — Phase 02 enables this test.")

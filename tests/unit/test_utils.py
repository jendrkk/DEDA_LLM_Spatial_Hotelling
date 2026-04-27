"""Unit tests for hotelling.utils – seeding and logging."""
from __future__ import annotations

import hotelling.utils.seeding as seeding_mod
import hotelling.utils.logging as logging_mod


class TestSeeding:
    def test_make_rng_returns_generator(self):
        import numpy as np

        rng = seeding_mod.make_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_none_seed(self):
        import numpy as np

        rng = seeding_mod.make_rng(None)
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_reproducible(self):
        rng1 = seeding_mod.make_rng(123)
        rng2 = seeding_mod.make_rng(123)
        import numpy as np

        a = rng1.random(10)
        b = rng2.random(10)
        np.testing.assert_array_equal(a, b)

    def test_derive_seed_deterministic(self):
        s1 = seeding_mod.derive_seed(42, "agent_A", "run_01")
        s2 = seeding_mod.derive_seed(42, "agent_A", "run_01")
        assert s1 == s2

    def test_derive_seed_different_labels(self):
        s1 = seeding_mod.derive_seed(42, "agent_A")
        s2 = seeding_mod.derive_seed(42, "agent_B")
        assert s1 != s2

    def test_derive_seed_in_range(self):
        seed = seeding_mod.derive_seed(0, "test")
        assert 0 <= seed < 2**31


class TestLogging:
    def test_get_logger_returns_logger(self):
        import logging

        logger = logging_mod.get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_has_handlers(self):
        logger = logging_mod.get_logger("test_module_2")
        assert len(logger.handlers) >= 1

    def test_setup_logging_no_error(self):
        logging_mod.setup_logging(level="WARNING")

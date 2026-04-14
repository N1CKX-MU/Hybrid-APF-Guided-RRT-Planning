from planner import config

def test_config_loads_defaults():
    assert hasattr(config, "GOAL_BIAS")
    assert hasattr(config, "APF_BLEND")
    assert hasattr(config, "MAX_ITER_DEFAULT")
    assert hasattr(config, "MAX_ITER_BENCHMARK")

def test_config_values_are_valid_types():
    assert isinstance(config.GOAL_BIAS, float)
    assert isinstance(config.MAX_ITER_BENCHMARK, int)
    assert config.MAX_ITER_BENCHMARK > 0
    assert config.GOAL_BIAS > 0.0

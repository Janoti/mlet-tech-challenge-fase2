from recsys.pipeline.params import load_params


def test_load_params_reads_sections():
    params = load_params("params.yaml")
    assert params["generate"]["seed"] == 42
    assert params["preprocess"]["test_size"] == 0.2
    assert params["evaluate"]["top_k"] == 10

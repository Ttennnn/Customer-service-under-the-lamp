from scripts.ch10.export_onnx import compare_predictions


def test_compare_predictions_requires_identical_labels():
    torch_probs = [[0.9, 0.1], [0.2, 0.8]]
    onnx_probs = [[0.9, 0.1], [0.2, 0.8]]
    assert (
        compare_predictions(torch_probs, onnx_probs, 0.5, atol=1e-4)["ok"]
        is True
    )

    bad = [[0.9, 0.1], [0.6, 0.4]]
    assert (
        compare_predictions(torch_probs, bad, 0.5, atol=1e-4)["ok"]
        is False
    )

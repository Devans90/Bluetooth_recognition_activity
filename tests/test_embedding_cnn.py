import torch

from src.bluetooth_embedding_cnn import HeadlessSimpleCNN


def test_headless_cnn_returns_embedding_vector():
    model = HeadlessSimpleCNN(in_channels=2)
    x = torch.randn(4, 2, 64, 64)
    emb = model(x)

    assert emb.shape == (4, 128)
    assert torch.isfinite(emb).all()


def test_headless_cnn_train_step_runs_without_error():
    model = HeadlessSimpleCNN(in_channels=2)
    x = torch.randn(8, 2, 64, 64)
    y = torch.randint(0, 5, (8,))

    logits = model.classifier(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    assert logits.shape == (8, 5)
    assert torch.isfinite(loss)

import pytest
import torch

@pytest.fixture(scope="session")
def batch():
    BATCH_SIZE = 4
    SEQ_LEN = 36
    N_VARS = 8

    torch.manual_seed(0)

    return torch.rand(BATCH_SIZE, SEQ_LEN, N_VARS)
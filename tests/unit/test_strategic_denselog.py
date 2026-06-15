import numpy as np
from pathlib import Path

from hotelling.simulation.dense_log import DenseLog


def test_denselog_write_and_reload(tmp_path: Path):
    N, T = 3, 10
    dl = DenseLog(run_dir=tmp_path, T=T, N=N, agent_ids=["0", "1", "2"],
                  price_grid=np.linspace(20, 55, 25), effort_grid=np.array([0.0]),
                  store_demand_profit=True, float_dtype="float32", dense_stride=1)
    for t in range(T):
        dl.write_step(t, np.zeros(N, int), np.zeros(N, int),
                      np.ones(N), np.full(N, 2.0))
    dl.flush()
    loaded = DenseLog.load(tmp_path)
    assert loaded.price_idx.shape == (T, N)
    assert (tmp_path / "price_grid.npy").exists()


from functools import partial

from src.slm.preprocess.stages.canonical import transform_canonicalize_row


if __name__ == "__main__":
    import ray 

    ray.init()

    ds = ray.data.read_parquet()
    canon_row_fn = partial(transform_canonicalize_row, canon_cfg=cfg.canonicalize)
    ds = ds.map(canon_row_fn)
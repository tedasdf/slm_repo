import ray

ray.init()

ds = ray.data.read_json(
    "./artifacts/data/dolma_v1_6_sample/",
    lines=True,
)

ds = ds.select_columns(["text"])

ds.write_parquet(
    "s3://attention-ntp-artifact/datasets/dolma_v1_6_parquet/"
)
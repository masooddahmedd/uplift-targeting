# Downloads the Criteo Uplift Prediction dataset v2.1 (Diemert et al., 2018) from Criteo's
# Hugging Face page and writes data/criteo.parquet with compact dtypes. ~14M rows from a randomized
# experiment: `treatment` = the user was eligible to be shown the incentive (ads), `visit` and
# `conversion` are the outcomes, f0..f11 are anonymised user features. Licence: CC BY-NC-SA 4.0,
# so the data is not redistributed with this repo.
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
DATA = Path(__file__).parent / "data"
RAW = DATA / "criteo-research-uplift-v2.1.csv.gz"
OUT = DATA / "criteo.parquet"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    if not RAW.exists():
        print("downloading", URL)
        tmp = RAW.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(RAW)
    dtypes = {f"f{i}": "float32" for i in range(12)}
    dtypes.update({c: "int8" for c in ("treatment", "conversion", "visit", "exposure")})
    df = pd.read_csv(RAW, dtype=dtypes)
    df.to_parquet(OUT, index=False)
    print(len(df), "rows ->", OUT)
    print(df.describe().T[["mean", "min", "max"]])


if __name__ == "__main__":
    main()

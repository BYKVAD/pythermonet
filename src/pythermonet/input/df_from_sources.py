import pandas as pd
from pathlib import Path

def df_from_csv(path: Path | str, sep: str = ';') -> pd.DataFrame:
    """
    Generates a dataframe from a csv.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pipe catalogue not found: {p}")

    df = pd.read_csv(p, sep=sep, engine="python")
    return df
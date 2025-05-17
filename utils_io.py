import os, glob, pandas as pd

def iter_files(root, suffix, strict_suffix=True):
    """Yield absolute paths of files with the given suffix (recursive)."""
    for path in glob.iglob(os.path.join(root, "**", "*"), recursive=True):
        if (path.endswith(suffix) if strict_suffix else suffix in path) and os.path.isfile(path):
            yield path

def save_df(df, path, **to_csv_kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, **to_csv_kw)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

"""Three pieces of setup work outside training that change nothing they produce.

- TRL builds `importlib.metadata.packages_distributions()` when it is imported: a walk over every
  installed distribution's file list, about 2.7 s on Kaggle's image of a thousand packages.
  transformers made the same call when the harness imported it, in the same process, with nothing
  installed in between (check 2 verifies that no package changes during a run), so TRL gets a copy
  of transformers' map for that one call. Nothing else about TRL's import changes.
- For every `Dataset.map`, `datasets` hashes the function and its arguments with dill to name the
  result's cache file (about 2 s over TRL's maps on Kaggle's CPU). A dataset that lives in memory
  has no cache files, so the name is never used: such maps get a random fingerprint, what
  `datasets` itself falls back to when hashing fails. It comes from `datasets`' own generator, not
  Python's global one, so no random draw of the run moves. Datasets backed by files keep hashing.
- bitsandbytes builds its 16-entry NF4 table on the GPU from a Python list for every weight it
  quantizes (`get_4bit_type`: a host-to-device copy and a normalization, each waiting for the
  GPU), 0.9 s over the model's 196 weights on Kaggle. The table is built once per device and each
  weight gets its own copy of it, as before.

The first two were found by a teammate's timing of the run's setup, the third by ours
(dev/timeline_probe.py).
"""

import importlib.metadata
import sys


def share_package_map():
    if "trl" in sys.modules:
        return
    tf = sys.modules.get("transformers.utils.import_utils")
    mapping = getattr(tf, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if not isinstance(mapping, dict) or not mapping:
        return
    original = importlib.metadata.packages_distributions

    def once():  # TRL's one call at its import, then the original again
        importlib.metadata.packages_distributions = original
        return {key: list(value) for key, value in mapping.items()}

    importlib.metadata.packages_distributions = once
    try:
        import trl.import_utils  # noqa: F401
    finally:
        importlib.metadata.packages_distributions = original


def patch_dataset_map():
    try:
        from datasets import Dataset
        from datasets.fingerprint import generate_random_fingerprint
    except Exception:  # noqa: BLE001 - datasets not importable: nothing to patch
        return
    original = Dataset.map
    if getattr(original, "_soupk", False):
        return

    def map(self, *args, **kwargs):  # noqa: A001 - the method it wraps
        if not self.cache_files and kwargs.get("new_fingerprint") is None:
            kwargs["new_fingerprint"] = generate_random_fingerprint()
        return original(self, *args, **kwargs)

    map._soupk = True
    map.__wrapped__ = original
    map.__doc__ = original.__doc__
    Dataset.map = map


def cache_nf4_table():
    try:
        import bitsandbytes.functional as bnb_functional
    except Exception:  # noqa: BLE001 - bitsandbytes not importable: nothing to patch
        return
    original = bnb_functional.get_4bit_type
    if getattr(original, "_soupk", False):
        return
    tables = {}

    def get_4bit_type(typename, device=None, blocksize=64):
        key = (typename, str(device), blocksize)
        if key not in tables:
            tables[key] = original(typename, device=device, blocksize=blocksize)
        return tables[key].clone()  # its own tensor per quant state, as each call made one

    get_4bit_type._soupk = True
    get_4bit_type.__wrapped__ = original
    bnb_functional.get_4bit_type = get_4bit_type

"""Explicit HuggingFace authentication for the gated ai4bharat/BPCC dataset.

The token is read from (in order of precedence):

1. the `HF_TOKEN` environment variable, if set;
2. the cache written by `huggingface-cli login` (read through huggingface_hub).

If neither is available, a clear error is raised. ai4bharat/BPCC and the IN22
sets are gated: request access on their Hugging Face pages first. Never put a
token in this repository.
"""

import os


def get_hf_token() -> str:
    """Return the HuggingFace token, or raise a clear error if none is found."""
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    try:
        from huggingface_hub import get_token
        cached = get_token()
    except Exception:  # huggingface_hub missing or too old
        cached = None
    if cached:
        return cached

    raise RuntimeError(
        "No HuggingFace token found. Set the HF_TOKEN environment variable, or run "
        "`huggingface-cli login`. ai4bharat/BPCC is a gated dataset and requires an "
        "authenticated, access-approved token."
    )


def ensure_hf_login() -> None:
    """Make the HF token available to `datasets.load_dataset` (via env var) and
    also attempt an explicit huggingface_hub login for good measure."""
    token = get_hf_token()
    os.environ.setdefault("HF_TOKEN", token)
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - best-effort convenience only
        print(f"[auth] huggingface_hub.login() skipped/failed ({exc}); relying on HF_TOKEN env var.")

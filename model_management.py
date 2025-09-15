# model_management.py
"""
Model management utilities for MagentaRT API.

This module handles checkpoint discovery, asset loading, and model selection logic.
It is designed to work with the global state managed in app.py without interfering
with the critical JAX/XLA initialization sequence.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Union, Literal, Tuple, List
import tarfile

import numpy as np
from pydantic import BaseModel
from huggingface_hub import snapshot_download, HfApi, hf_hub_download


# ---- Constants and Patterns ----
_FINETUNE_REPO_DEFAULT = os.getenv("MRT_ASSETS_REPO", "thepatch/magenta-ft")
_STEP_RE = re.compile(r"(?:^|/)checkpoint_(\d+)(?:/|\.tar\.gz|\.tgz)?$")


# ---- Pydantic Models ----
class ModelSelect(BaseModel):
    size: Optional[Literal["base","large"]] = None
    repo_id: Optional[str] = None
    revision: Optional[str] = "main"
    step: Optional[Union[int, str]] = None   # allow "latest"
    assets_repo_id: Optional[str] = None     # default: follow repo_id
    sync_assets: bool = True                 # load mean/centroids from repo
    prewarm: bool = False                    # call get_mrt() to build right away
    stop_active: bool = True                 # auto-stop jams; else 409
    dry_run: bool = False                    # validate only, don't swap


# ---- Checkpoint Discovery ----
class CheckpointManager:
    """Handles checkpoint discovery and validation without modifying global state."""
    
    @staticmethod
    def list_ckpt_steps(repo_id: str, revision: str = "main") -> List[int]:
        """
        List available checkpoint steps in a HF model repo without downloading all weights.
        Looks for:
          checkpoint_<step>/
          checkpoint_<step>.tgz | .tar.gz
          archives/checkpoint_<step>.tgz | .tar.gz
        """
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
        steps = set()
        for f in files:
            m = _STEP_RE.search(f)
            if m:
                try:
                    steps.add(int(m.group(1)))
                except:
                    pass
        return sorted(steps)

    @staticmethod
    def step_exists(repo_id: str, revision: str, step: int) -> bool:
        """Check if a specific checkpoint step exists in the repo."""
        return step in CheckpointManager.list_ckpt_steps(repo_id, revision)

    @staticmethod
    def resolve_checkpoint_dir() -> Optional[str]:
        """
        Resolve the checkpoint directory from environment variables.
        Downloads and extracts if necessary.
        Returns the path to the checkpoint directory or None if not configured.
        """
        repo_id = os.getenv("MRT_CKPT_REPO")
        if not repo_id:
            return None
        step = os.getenv("MRT_CKPT_STEP")  # e.g. "1863001"

        root = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=os.getenv("MRT_CKPT_REV", "main"),
            local_dir="/home/appuser/.cache/mrt_ckpt/repo",
            local_dir_use_symlinks=False,
        ))

        # Prefer an archive if present (more reliable for Zarr/T5X)
        arch_names = [
            f"checkpoint_{step}.tgz",
            f"checkpoint_{step}.tar.gz",
            f"archives/checkpoint_{step}.tgz",
            f"archives/checkpoint_{step}.tar.gz",
        ] if step else []

        cache_root = Path("/home/appuser/.cache/mrt_ckpt/extracted")
        cache_root.mkdir(parents=True, exist_ok=True)
        for name in arch_names:
            arch = root / name
            if arch.is_file():
                out_dir = cache_root / f"checkpoint_{step}"
                marker = out_dir.with_suffix(".ok")
                if not marker.exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(arch, "r:*") as tf:
                        tf.extractall(out_dir)
                    marker.write_text("ok")
                # sanity: require .zarray to exist inside the extracted tree
                if not any(out_dir.rglob(".zarray")):
                    raise RuntimeError(f"Extracted archive missing .zarray files: {out_dir}")
                return str(out_dir / f"checkpoint_{step}") if (out_dir / f"checkpoint_{step}").exists() else str(out_dir)

        # No archive; try raw folder from repo and sanity check.
        if step:
            raw = root / f"checkpoint_{step}"
            if raw.is_dir():
                if not any(raw.rglob(".zarray")):
                    raise RuntimeError(
                        f"Downloaded checkpoint_{step} appears incomplete (no .zarray). "
                        "Upload as a .tgz or push via git from a Unix shell."
                    )
                return str(raw)

        # Pick latest if no step
        step_dirs = [d for d in root.iterdir() if d.is_dir() and re.match(r"checkpoint_\d+$", d.name)]
        if step_dirs:
            pick = max(step_dirs, key=lambda d: int(d.name.split('_')[-1]))
            if not any(pick.rglob(".zarray")):
                raise RuntimeError(f"Downloaded {pick} appears incomplete (no .zarray).")
            return str(pick)

        return None


# ---- Asset Management ----
class AssetManager:
    """
    Handles finetune asset loading and management.
    
    This class modifies global variables in the calling module, but encapsulates
    the logic for loading and validating assets.
    """
    
    def __init__(self):
        # These will be set by the calling module
        self.mean_embed = None
        self.centroids = None
        self.assets_repo_id = None
    
    def load_finetune_assets_from_hf(self, repo_id: Optional[str], mrt=None) -> Tuple[bool, str]:
        """
        Download & load mean_style_embed.npy and cluster_centroids.npy from a HF model repo.
        Safe to call multiple times; will overwrite instance vars if successful.
        
        Args:
            repo_id: HuggingFace repo ID, defaults to _FINETUNE_REPO_DEFAULT
            mrt: MagentaRT instance for dimension validation (optional)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        repo_id = repo_id or _FINETUNE_REPO_DEFAULT
        try:
            mean_path = None
            cent_path = None
            try:
                mean_path = hf_hub_download(repo_id, filename="mean_style_embed.npy", repo_type="model")
            except Exception:
                pass
            try:
                cent_path = hf_hub_download(repo_id, filename="cluster_centroids.npy", repo_type="model")
            except Exception:
                pass

            if mean_path is None and cent_path is None:
                return False, f"No finetune asset files found in repo {repo_id}"

            if mean_path is not None:
                m = np.load(mean_path)
                if m.ndim != 1:
                    return False, f"mean_style_embed.npy must be 1-D (got {m.shape})"
            else:
                m = None

            if cent_path is not None:
                c = np.load(cent_path)
                if c.ndim != 2:
                    return False, f"cluster_centroids.npy must be 2-D (got {c.shape})"
            else:
                c = None

            # Optional: shape check vs model embedding dim once model is alive
            if mrt is not None:
                try:
                    d = int(mrt.style_model.config.embedding_dim)
                    if m is not None and m.shape[0] != d:
                        return False, f"mean_style_embed dim {m.shape[0]} != model dim {d}"
                    if c is not None and c.shape[1] != d:
                        return False, f"cluster_centroids dim {c.shape[1]} != model dim {d}"
                except Exception:
                    # Model not built yet; we'll trust the files and rely on runtime checks later
                    pass

            # Update instance variables
            self.mean_embed = m.astype(np.float32, copy=False) if m is not None else None
            self.centroids = c.astype(np.float32, copy=False) if c is not None else None
            self.assets_repo_id = repo_id
            
            logging.info("Loaded finetune assets from %s (mean=%s, centroids=%s)",
                         repo_id,
                         "yes" if self.mean_embed is not None else "no",
                         f"{self.centroids.shape[0]}x{self.centroids.shape[1]}" if self.centroids is not None else "no")
            return True, "ok"
        except Exception as e:
            logging.exception("Failed to load finetune assets: %s", e)
            return False, str(e)

    def ensure_assets_loaded(self, mrt=None):
        """Best-effort lazy load if nothing is loaded yet."""
        if self.mean_embed is None and self.centroids is None:
            self.load_finetune_assets_from_hf(self.assets_repo_id or _FINETUNE_REPO_DEFAULT, mrt)

    def get_status(self, mrt=None) -> dict:
        """Get current asset status."""
        d = None
        if mrt is not None:
            try:
                d = int(mrt.style_model.config.embedding_dim)
            except Exception:
                pass
        
        return {
            "repo_id": self.assets_repo_id,
            "mean_loaded": self.mean_embed is not None,
            "centroids_loaded": self.centroids is not None,
            "centroid_count": None if self.centroids is None else int(self.centroids.shape[0]),
            "embedding_dim": d,
        }


# ---- Model Selection Logic ----
class ModelSelector:
    """
    Handles model selection and validation logic.
    
    This class encapsulates the complex logic from the /model/select endpoint
    while keeping environment variable management in the calling code.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager, asset_manager: AssetManager):
        self.checkpoint_manager = checkpoint_manager
        self.asset_manager = asset_manager
    
    def validate_selection(self, req: ModelSelect) -> Tuple[bool, dict]:
        """
        Validate a model selection request without making any changes.
        
        Returns:
            Tuple of (success: bool, result_dict: dict)
        """
        # Current env defaults
        cur = {
            "size": os.getenv("MRT_SIZE", "large"),
            "repo": os.getenv("MRT_CKPT_REPO"),
            "rev": os.getenv("MRT_CKPT_REV", "main"),
            "step": os.getenv("MRT_CKPT_STEP"),
            "assets": os.getenv("MRT_ASSETS_REPO", _FINETUNE_REPO_DEFAULT),
        }

        # Flags for special step values
        no_ckpt = isinstance(req.step, str) and req.step.lower() == "none"
        latest = isinstance(req.step, str) and req.step.lower() == "latest"

        # Target selection
        tgt = {
            "size": req.size or cur["size"],
            "repo": None if no_ckpt else (req.repo_id or cur["repo"]),
            "rev": req.revision if req.revision is not None else cur["rev"],
            "step": None if (no_ckpt or latest) else (str(req.step) if req.step is not None else cur["step"]),
            "assets": req.assets_repo_id or req.repo_id or cur["assets"],
        }

        # Case 1: No checkpoint (stock model)
        if no_ckpt:
            return True, {
                "target_size": tgt["size"],
                "target_repo": None,
                "target_revision": None,
                "target_step": None,
                "assets_repo": None,
                "assets_probe": {"ok": True, "message": "skipped"},
            }

        # Case 2: Checkpoint selection
        if not tgt["repo"]:
            return False, {"error": "repo_id is required for model selection."}

        # Enumerate available steps
        try:
            steps = self.checkpoint_manager.list_ckpt_steps(tgt["repo"], tgt["rev"])
        except Exception as e:
            return False, {"error": f"Failed to list checkpoints: {e}"}
        
        if not steps:
            return False, {
                "error": f"No checkpoint files found in {tgt['repo']}@{tgt['rev']}", 
                "discovered_steps": steps
            }

        # Choose step (explicit or latest)
        chosen_step = int(tgt["step"]) if tgt["step"] is not None else steps[-1]
        if chosen_step not in steps:
            return False, {
                "error": f"checkpoint_{chosen_step} not present in {tgt['repo']}@{tgt['rev']}", 
                "discovered_steps": steps
            }

        # Optional finetune assets probe
        assets_ok, assets_msg = True, "skipped"
        if req.sync_assets:
            try:
                api = HfApi()
                files = set(api.list_repo_files(repo_id=tgt["assets"], repo_type="model"))
                if ("mean_style_embed.npy" not in files) and ("cluster_centroids.npy" not in files):
                    assets_ok, assets_msg = False, f"No finetune asset files in {tgt['assets']}"
                else:
                    assets_msg = "found"
            except Exception as e:
                assets_ok, assets_msg = False, f"probe failed: {e}"

        return True, {
            "target_size": tgt["size"],
            "target_repo": tgt["repo"],
            "target_revision": tgt["rev"],
            "target_step": chosen_step,
            "assets_repo": tgt["assets"] if req.sync_assets else None,
            "assets_probe": {"ok": assets_ok, "message": assets_msg},
        }

    def prepare_env_changes(self, req: ModelSelect, validation_result: dict) -> dict:
        """
        Prepare the environment variable changes needed for a model selection.
        
        Args:
            req: The model selection request
            validation_result: Result from validate_selection()
            
        Returns:
            Dictionary of environment variable changes to apply
        """
        no_ckpt = isinstance(req.step, str) and req.step.lower() == "none"
        
        if no_ckpt:
            # Clear checkpoint env vars for stock model
            return {
                "MRT_SIZE": validation_result["target_size"],
                "MRT_CKPT_REPO": None,  # None means delete the env var
                "MRT_CKPT_REV": None,
                "MRT_CKPT_STEP": None,
                "MRT_ASSETS_REPO": None,
            }
        else:
            # Set checkpoint env vars
            env_changes = {
                "MRT_SIZE": validation_result["target_size"],
                "MRT_CKPT_REPO": validation_result["target_repo"],
                "MRT_CKPT_REV": validation_result["target_revision"],
                "MRT_CKPT_STEP": str(validation_result["target_step"]),
            }
            if req.sync_assets:
                env_changes["MRT_ASSETS_REPO"] = validation_result["assets_repo"]
            return env_changes
# boundary_detector_tool.py

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# ✅ Import the trained model implementation from models.py (future-proof for more models)
from .models import FunctionBoundaryModel


# ---------------------------
# Pydantic schema for tool I/O
# ---------------------------
class BoundaryDetectorArgs(BaseModel):
    model_path: Optional[str] = Field(
        default="models/function_boundary.pth",
        description="Checkpoint path for the boundary model."
    )
    # Parity with infer.py (defaults mirror your infer script behavior)
    min_function_size: int = Field(
        default=0, ge=0,
        description="Minimum number of bytes for a multi-byte function (0 matches infer default)."
    )
    merge_distance: int = Field(
        default=0, ge=0,
        description="Merge nearby functions within this byte distance (0 = disabled, matches infer default)."
    )
    resolve_overlaps: bool = Field(
        default=True,
        description="When True, resolve overlapping functions using confidences (matches infer)."
    )
    stride: int = Field(
        default=2048, ge=1,
        description="Sliding-window stride in bytes (matches infer default)."
    )


# ---------------------------
# Tool implementation
# ---------------------------
class BoundaryDetectorTool:
    name = "boundary_detector"
    description = "Detects function boundaries in binary code using a trained Mamba model."
    ArgsModel = BoundaryDetectorArgs

    @classmethod
    def tool_spec(cls) -> Dict[str, Any]:
        """Return function-calling schema derived from Pydantic."""
        try:
            params = cls.ArgsModel.model_json_schema()  # Pydantic v2
        except AttributeError:
            params = cls.ArgsModel.schema()             # Pydantic v1 fallback
        params.setdefault("additionalProperties", False)
        return {"name": cls.name, "description": cls.description.strip(), "parameters": params}

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        min_function_size: int = 0,
        merge_distance: int = 0,
        resolve_overlaps: bool = True,
        stride: int = 2048,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[FunctionBoundaryModel] = None

        # Sliding window params (keep stride explicit like infer.py)
        self.window_size: int = 4096  # will be updated from ckpt config
        self.stride: int = int(stride)

        # Post-processing knobs (parity with infer.py)
        self.min_function_size = int(min_function_size)
        self.merge_distance = int(merge_distance)
        self.resolve_overlaps = bool(resolve_overlaps)

        if model_path:
            self.load_model(model_path)

    # ---------------------------
    # Model loading (parity with infer.py)
    # ---------------------------
    def load_model(self, model_path: str):
        """Load a trained checkpoint and instantiate the exact same architecture."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        # Pull config written by training script
        cfg = ckpt.get("config", {}) or {}

        # window_size must match checkpoint for positional embeddings
        self.window_size = int(cfg.get("window_size", self.window_size))
        # IMPORTANT: keep stride as the user-specified value (like infer.py).
        # Do NOT force override from ckpt, or you’ll drift from infer if it uses a different stride.

        # Build the model with the same hyperparameters saved at train time
        self.model = FunctionBoundaryModel(
            window_size=self.window_size,
            embed_dim=cfg.get("embed_dim", 256),
            mamba_dim=cfg.get("mamba_dim", 512),
            n_layers=cfg.get("n_layers", 4),
            dropout=0.0,         # no dropout at inference (same as infer.py)
            n_classes=5,
        )

        # Load state dict strictly to catch accidental architecture drift
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # ---------------------------
    # Inference + post-processing (identical to infer.py)
    # ---------------------------
    def run(self, text_array: np.ndarray, base_address: int) -> Dict[str, Any]:
        """
        Execute boundary detection.

        Inputs:
          - text_array: np.ndarray of uint8/ints in [0,255], length = .text size
          - base_address: int base virtual address for the text section

        Returns:
          - dict with fields: success, functions[], total_functions, message
        """
        if self.model is None:
            return {"error": "Model not loaded", "success": False}

        try:
            text_len = int(len(text_array))
            n_classes = 5

            # Accumulate probabilities per byte via sliding windows (same as infer.py)
            predictions = np.zeros((text_len, n_classes), dtype=np.float32)
            counts = np.zeros(text_len, dtype=np.int32)

            with torch.no_grad():
                for start in range(0, text_len, self.stride):
                    end = min(start + self.window_size, text_len)
                    window_len = end - start

                    # Pad last window like infer.py
                    if window_len < self.window_size:
                        window = np.pad(
                            text_array[start:end],
                            (0, self.window_size - window_len),
                            constant_values=0,
                        )
                    else:
                        window = text_array[start:end]

                    window_tensor = torch.from_numpy(window.astype(np.int64)).unsqueeze(0).to(self.device)

                    # Model forward → logits → softmax probabilities
                    logits = self.model(window_tensor)  # [1, window_size, 5]
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [window_size, 5]

                    # Accumulate only the valid portion
                    predictions[start:start + window_len] += probs[:window_len]
                    counts[start:start + window_len] += 1

            # Average overlapping predictions
            predictions = predictions / np.maximum(counts[:, np.newaxis], 1)

            # Argmax labels per byte (matches infer.py)
            class_predictions = np.argmax(predictions, axis=1)

            # ---------- Post-processing (exactly like infer.py) ----------
            functions = extract_functions(
                predictions=class_predictions,
                probabilities=predictions if self.resolve_overlaps else None,
                base_addr=base_address,
                min_function_size=self.min_function_size,
            )

            if self.resolve_overlaps:
                functions = resolve_overlapping_functions(functions)

            if self.merge_distance > 0:
                functions = merge_nearby_functions(functions, self.merge_distance)

            functions.sort(key=lambda x: x["start_offset"])

            return {
                "success": True,
                "functions": functions,
                "total_functions": len(functions),
                "message": f"Detected {len(functions)} functions",
            }

        except Exception as e:
            return {"error": f"Boundary detection failed: {e}", "success": False}


# ---------------------------
# Helpers copied from infer.py (kept here for 1:1 parity)
# ---------------------------
def extract_functions(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    base_addr: int,
    min_function_size: int = 0,
) -> List[Dict]:
    """
    Extract function boundaries from per-byte class predictions.

    predictions: [N] integers in {0:none,1:start,2:body,3:end,4:single}
    probabilities: optional [N,5] softmax probs for overlap resolution & confidence
    """
    functions: List[Dict] = []

    # Singles (label 4)
    single_positions = np.where(predictions == 4)[0]
    for pos in single_positions:
        func = {
            "start_offset": int(pos),
            "end_offset": int(pos),
            "size": 1,
            "start_address": int(base_addr + pos),
            "end_address": int(base_addr + pos + 1),
            "type": "single",
        }
        func["confidence"] = float(probabilities[pos, 4]) if probabilities is not None else 1.0
        functions.append(func)

    # Multi-byte: start=1, end=3
    start_positions = np.where(predictions == 1)[0]
    for start_pos in start_positions:
        remaining = predictions[start_pos + 1 :]
        end_positions_in_remaining = np.where(remaining == 3)[0]
        if len(end_positions_in_remaining) == 0:
            continue

        end_pos = start_pos + 1 + int(end_positions_in_remaining[0])
        size = end_pos - start_pos + 1
        if size < min_function_size:
            continue

        func = {
            "start_offset": int(start_pos),
            "end_offset": int(end_pos),
            "size": int(size),
            "start_address": int(base_addr + start_pos),
            "end_address": int(base_addr + end_pos + 1),
            "type": "normal",
        }

        if probabilities is not None:
            start_conf = float(probabilities[start_pos, 1])
            end_conf = float(probabilities[end_pos, 3])
            func["confidence"] = (start_conf + end_conf) / 2.0
        else:
            func["confidence"] = 1.0

        functions.append(func)

    # Sort by start for deterministic downstream handling (infer.py does this)
    functions.sort(key=lambda x: x["start_offset"])
    return functions


def resolve_overlapping_functions(functions: List[Dict]) -> List[Dict]:
    """Resolve overlaps by keeping the higher-confidence candidate (infer.py behavior)."""
    if not functions:
        return functions

    # Sort by start, then by confidence desc
    functions.sort(key=lambda x: (x["start_offset"], -x.get("confidence", 0.0)))

    non_overlapping: List[Dict] = []
    for func in functions:
        overlaps = False
        for selected in list(non_overlapping):  # iterate over a snapshot since we might remove
            if not (
                func["end_offset"] < selected["start_offset"]
                or func["start_offset"] > selected["end_offset"]
            ):
                # Overlap → keep higher confidence
                if func.get("confidence", 0.0) > selected.get("confidence", 0.0):
                    non_overlapping.remove(selected)
                else:
                    overlaps = True
                    break
        if not overlaps:
            non_overlapping.append(func)

    non_overlapping.sort(key=lambda x: x["start_offset"])
    return non_overlapping


def merge_nearby_functions(functions: List[Dict], merge_distance: int = 16) -> List[Dict]:
    """Optionally merge very close normal functions (same as infer.py)."""
    if not functions:
        return functions

    single_funcs = [f for f in functions if f.get("type") == "single"]
    normal_funcs = [f for f in functions if f.get("type") != "single"]

    if not normal_funcs:
        return functions

    merged: List[Dict] = []
    current = normal_funcs[0].copy()

    for func in normal_funcs[1:]:
        gap = func["start_offset"] - current["end_offset"]
        if gap <= merge_distance:
            # Merge into current
            current["end_offset"] = func["end_offset"]
            current["end_address"] = func["end_address"]
            current["size"] = current["end_offset"] - current["start_offset"] + 1
            if "confidence" in current and "confidence" in func:
                current["confidence"] = (current["confidence"] + func["confidence"]) / 2.0
        else:
            merged.append(current)
            current = func.copy()

    merged.append(current)

    # Reattach singles and sort
    merged.extend(single_funcs)
    merged.sort(key=lambda x: x["start_offset"])
    return merged

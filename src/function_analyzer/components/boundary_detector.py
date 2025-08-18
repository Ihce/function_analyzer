# boundary_detector_tool.py  (or wherever the tool lives)

import torch
import numpy as np
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# ✅ Use the exact trained model implementation
from .models import FunctionBoundaryModel


class BoundaryDetectorArgs(BaseModel):
    model_path: Optional[str] = Field(
        default="models/phase1/best_model.pth",
        description="Checkpoint path for the boundary model."
    )


class BoundaryDetectorTool:
    name = "boundary_detector"
    description = "Detects function boundaries using the trained Mamba model."
    ArgsModel = BoundaryDetectorArgs

    @classmethod
    def tool_spec(cls) -> Dict[str, Any]:
        try:
            params = cls.ArgsModel.model_json_schema()
        except AttributeError:
            params = cls.ArgsModel.schema()
        params.setdefault("additionalProperties", False)
        return {"name": cls.name, "description": cls.description.strip(), "parameters": params}

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.window_size = 4096
        self.stride = 2048
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        cfg = ckpt.get("config", {}) or {}
        # enforce exact architecture used in training
        self.window_size = int(cfg.get("window_size", self.window_size))
        self.stride      = int(cfg.get("stride", self.stride))

        self.model = FunctionBoundaryModel(
            window_size=self.window_size,
            embed_dim=cfg.get("embed_dim", 256),
            mamba_dim=cfg.get("mamba_dim", 512),
            n_layers=cfg.get("n_layers", 4),
            dropout=0.0,  # no dropout at inference
            n_classes=5,
        )

        # Load strictly so we catch accidental drifts
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def run(self, text_array: np.ndarray, base_address: int) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not loaded", "success": False}

        try:
            text_len = int(len(text_array))
            n_classes = 5
            preds = np.zeros((text_len, n_classes), dtype=np.float32)
            counts = np.zeros(text_len, dtype=np.int32)

            with torch.no_grad():
                for start in range(0, text_len, self.stride):
                    end = min(start + self.window_size, text_len)
                    window_len = end - start

                    window = (np.pad(text_array[start:end],
                                     (0, self.window_size - window_len),
                                     constant_values=0)
                              if window_len < self.window_size else
                              text_array[start:end])

                    x = torch.from_numpy(window.astype(np.int64)).unsqueeze(0).to(self.device)
                    # model expects shape [B, L]; your forward embeds bytes, adds pos, then Mamba
                    logits = self.model(x)                         # [1, window_size, 5]
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                    preds[start:start + window_len] += probs[:window_len]
                    counts[start:start + window_len] += 1

            preds = preds / np.maximum(counts[:, None], 1)
            labels = np.argmax(preds, axis=1)

            # Build function list (same logic you had, plus confidences)
            functions = []

            # singles (label 4)
            for pos in np.where(labels == 4)[0]:
                functions.append({
                    "start_offset": int(pos),
                    "end_offset": int(pos),
                    "size": 1,
                    "start_address": int(base_address + pos),
                    "end_address": int(base_address + pos + 1),
                    "type": "single",
                    "confidence": float(preds[pos, 4]),
                })

            # multi: start=1, end=3
            for s in np.where(labels == 1)[0]:
                rem = labels[s + 1:]
                ends = np.where(rem == 3)[0]
                if len(ends) == 0:
                    continue
                e = s + 1 + int(ends[0])
                size = e - s + 1
                if size >= 16:
                    start_conf = float(preds[s, 1])
                    end_conf = float(preds[e, 3]) if e < len(preds) else 0.0
                    functions.append({
                        "start_offset": int(s),
                        "end_offset": int(e),
                        "size": int(size),
                        "start_address": int(base_address + s),
                        "end_address": int(base_address + e + 1),
                        "type": "normal",
                        "confidence": (start_conf + end_conf) / 2.0,
                    })

            functions.sort(key=lambda x: x["start_offset"])
            return {"success": True, "functions": functions,
                    "total_functions": len(functions),
                    "message": f"Detected {len(functions)} functions"}

        except Exception as e:
            return {"error": f"Boundary detection failed: {e}", "success": False}

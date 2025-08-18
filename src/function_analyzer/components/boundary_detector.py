"""Tool for detecting function boundaries using Mamba model."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from mamba_ssm import Mamba

# ✨ Pydantic (self-describing tool schema)
from pydantic import BaseModel, Field


class MambaBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class FunctionBoundaryModel(nn.Module):
    def __init__(self, window_size=4096):
        super().__init__()
        self.embedding = nn.Embedding(256, 256)
        self.input_proj = nn.Linear(256, 512)
        self.mamba_layers = nn.ModuleList([MambaBlock(512) for _ in range(4)])
        self.final_norm = nn.LayerNorm(512)
        self.output_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 classes: none, start, body, end, single
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output_proj(x)


# ✨ Args model for function-calling (optional model_path override)
class BoundaryDetectorArgs(BaseModel):
    model_path: Optional[str] = Field(
        default=None,
        description="Optional checkpoint override for the boundary model.",
    )


class BoundaryDetectorTool:
    """Detect function boundaries using trained Mamba model."""

    name = "boundary_detector"
    description = """Detects function boundaries in binary code using neural network.
    Input: text_array from binary_loader, model_path
    Output: list of detected functions with start/end offsets
    Use this after loading the binary to find where functions are."""

    # ✨ Tell the agent what args this tool expects
    ArgsModel = BoundaryDetectorArgs

    @classmethod
    def tool_spec(cls) -> Dict[str, Any]:
        """Return OpenAI/Ollama function-calling schema generated from Pydantic."""
        try:
            params = cls.ArgsModel.model_json_schema()  # Pydantic v2
        except AttributeError:
            params = cls.ArgsModel.schema()             # Pydantic v1 fallback
        params.setdefault("additionalProperties", False)
        return {
            "name": cls.name,
            "description": cls.description.strip(),
            "parameters": params,
        }

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.window_size = 4096
        self.stride = 2048

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = FunctionBoundaryModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def run(self, text_array: np.ndarray, base_address: int) -> Dict[str, Any]:
        """Execute the tool."""
        if self.model is None:
            return {"error": "Model not loaded", "success": False}

        try:
            # Sliding window prediction
            text_len = len(text_array)
            predictions = np.zeros((text_len, 5), dtype=np.float32)
            counts = np.zeros(text_len, dtype=np.int32)

            with torch.no_grad():
                for start in range(0, text_len, self.stride):
                    end = min(start + self.window_size, text_len)
                    window_len = end - start

                    # Prepare window
                    if window_len < self.window_size:
                        window = np.pad(text_array[start:end],
                                      (0, self.window_size - window_len),
                                      constant_values=0)
                    else:
                        window = text_array[start:end]

                    # Predict
                    window_tensor = torch.from_numpy(window.astype(np.int64)).unsqueeze(0).to(self.device)
                    logits = self.model(window_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                    # Accumulate
                    predictions[start:start + window_len] += probs[:window_len]
                    counts[start:start + window_len] += 1

            # Average and get class predictions
            predictions = predictions / np.maximum(counts[:, np.newaxis], 1)
            class_predictions = np.argmax(predictions, axis=1)

            # Extract functions
            functions = []

            # Find single-byte functions (class 4)
            single_positions = np.where(class_predictions == 4)[0]
            for pos in single_positions:
                functions.append({
                    "start_offset": int(pos),
                    "end_offset": int(pos),
                    "size": 1,
                    "start_address": int(base_address + pos),
                    "type": "single"
                })

            # Find multi-byte functions (start=1, end=3)
            start_positions = np.where(class_predictions == 1)[0]
            for start_pos in start_positions:
                remaining = class_predictions[start_pos + 1:]
                end_positions = np.where(remaining == 3)[0]

                if len(end_positions) > 0:
                    end_pos = start_pos + 1 + end_positions[0]
                    size = end_pos - start_pos + 1

                    if size >= 16:  # Min function size
                        functions.append({
                            "start_offset": int(start_pos),
                            "end_offset": int(end_pos),
                            "size": int(size),
                            "start_address": int(base_address + start_pos),
                            "type": "normal"
                        })

            functions.sort(key=lambda x: x["start_offset"])

            return {
                "success": True,
                "functions": functions,
                "total_functions": len(functions),
                "message": f"Detected {len(functions)} functions"
            }

        except Exception as e:
            return {"error": f"Boundary detection failed: {str(e)}", "success": False}

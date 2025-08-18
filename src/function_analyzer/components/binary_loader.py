"""Tool for loading PE binaries and extracting .text section."""

import pefile
import numpy as np
from pathlib import Path
from typing import Dict, Any

# ✨ Pydantic (self-describing tool schema)
from pydantic import BaseModel, Field


class BinaryLoaderArgs(BaseModel):
    file_path: str = Field(..., min_length=1, description="Path to the PE file on disk.")


class BinaryLoaderTool:
    name = "binary_loader"
    description = """Loads a PE binary file and extracts the .text section.
    Input: path to PE file
    Output: dictionary with text_array, base_address, and metadata
    Use this first to load any binary file."""

    ArgsModel = BinaryLoaderArgs

    @classmethod
    def tool_spec(cls) -> Dict[str, Any]:
        """Return OpenAI/Ollama function-calling schema generated from Pydantic."""
        try:
            params = cls.ArgsModel.model_json_schema()  # Pydantic v2
        except AttributeError:
            params = cls.ArgsModel.schema()             # Pydantic v1 fallback
        # Be strict about extra args
        params.setdefault("additionalProperties", False)
        return {
            "name": cls.name,
            "description": cls.description.strip(),
            "parameters": params,
        }

    def __init__(self):
        self.last_result = None

    def run(self, file_path: str) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            pe = pefile.PE(file_path)

            # Find .text section
            text_section = None
            for section in pe.sections:
                if b'.text' in section.Name:
                    text_section = section
                    break

            if not text_section:
                return {"error": "No .text section found in PE file"}

            # Extract text bytes
            text_data = text_section.get_data()
            text_array = np.frombuffer(text_data, dtype=np.uint8)

            # Get metadata
            image_base = pe.OPTIONAL_HEADER.ImageBase
            section_rva = text_section.VirtualAddress
            is_64 = pe.FILE_HEADER.Machine == 0x8664

            result = {
                "success": True,
                "file_path": file_path,
                "text_array": text_array,
                "image_base": image_base,
                "section_rva": section_rva,
                "base_address": image_base + section_rva,
                "architecture": "x64" if is_64 else "x86",
                "is_64bit": is_64,
                "text_size": len(text_array),
                "message": f"Successfully loaded PE file. Text section: {len(text_array)} bytes"
            }

            self.last_result = result
            pe.close()
            return result

        except Exception as e:
            return {"error": f"Failed to load binary: {str(e)}", "success": False}

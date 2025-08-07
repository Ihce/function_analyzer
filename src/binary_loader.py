"""Tool for loading PE binaries and extracting .text section."""

import pefile
import numpy as np
from pathlib import Path
from typing import Dict, Any


class BinaryLoaderTool:
    """Extract .text section from PE files."""

    name = "binary_loader"
    description = """Loads a PE binary file and extracts the .text section.
    Input: path to PE file
    Output: dictionary with text_array, base_address, and metadata
    Use this first to load any binary file."""

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

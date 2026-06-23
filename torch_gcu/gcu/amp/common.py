#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
import torch_gcu

__all__ = ["amp_definitely_not_available"]

def amp_definitely_not_available():
    return not torch_gcu.gcu.is_available()

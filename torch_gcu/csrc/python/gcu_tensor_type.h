#include <torch/csrc/utils/tensor_new.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

namespace tensors {

// Initializes the Python tensor type objects: torch.gcu.FloatTensor,
// torch.gcu.DoubleTensor, etc. and binds them in their containing modules.
void initialize_python_bindings();

TORCH_GCU_API PyMethodDef* gcu_tensor_type_functions();

}  // namespace tensors

}  // namespace torch_gcu

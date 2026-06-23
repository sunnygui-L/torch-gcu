file(GLOB _GCU_SRCS *.cpp *.h *.cc aot_ops/*.cpp aot_ops/*.cc aot_ops/*.h shape_inference/*.h shape_inference/*.cpp)

list(APPEND GCU_SRCS ${_GCU_SRCS})

# Pass to parent
set(GCU_SRCS ${GCU_SRCS} PARENT_SCOPE)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        # PATTERN "*.h"
        # PATTERN "*.hpp"
        PATTERN "gcu_macros.h"
        PATTERN "gcu_stream.h"
        PATTERN "gcu_event.h"
        PATTERN "gcu_guard.h"
        PATTERN "gcu_guard_impl.h"
        PATTERN "gcu_custom_api.h"
        PATTERN "gcu_generator_impl.h"
        PATTERN "philox_gcu_state_raw.h"
        PATTERN "gcu_utils.h")


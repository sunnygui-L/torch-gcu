
file(GLOB _DIST_SRCS *.cpp *.cc *.h)

list(APPEND DIST_SRCS ${_DIST_SRCS})

# Pass to parent
set(DIST_SRCS ${DIST_SRCS} PARENT_SCOPE)
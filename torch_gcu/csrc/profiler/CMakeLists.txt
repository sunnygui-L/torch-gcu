file(GLOB_RECURSE _PROF_SRCS *.cpp *.cc *.h)

list(APPEND PROF_SRCS ${_PROF_SRCS})

# Pass to parent
set(PROF_SRCS ${PROF_SRCS} PARENT_SCOPE)
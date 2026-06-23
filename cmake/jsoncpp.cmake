include(ExternalProject)
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")

set(JSONCPP_SOURCE_DIR /opt/jsoncpp)
if(NOT JSONCPP_TAG)
    set(JSONCPP_TAG 1.8.4)
endif()

set(JSONCPP_PREFIX_DIR  ${THIRD_PARTY_PATH}/jsoncpp-${JSONCPP_TAG} CACHE PATH "jsoncpp root directory" FORCE)
set(JSONCPP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/jsoncpp-${JSONCPP_TAG} CACHE PATH "jsoncpp install directory" FORCE)
set(JSONCPP_INCLUDE_DIR ${JSONCPP_INSTALL_DIR}/include)

unset(JSONCPP_LIBRARIES)
list(APPEND JSONCPP_LIBRARIES "${JSONCPP_INSTALL_DIR}/lib/libjsoncpp_static.a")
list(APPEND JSONCPP_LIBRARIES "${JSONCPP_INSTALL_DIR}/lib/libjsoncpp.so")

set(JSONCPP_COMMON_CMAKE_ARGS
    ${CMAKE_ARGS}
    -DCMAKE_INSTALL_PREFIX=${JSONCPP_INSTALL_DIR}
    -DJSONCPP_WITH_TESTS=OFF
    -DCMAKE_INSTALL_LIBDIR=lib
    "-DCMAKE_CXX_FLAGS=-fPIC -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}"
)

ExternalProject_Add(
    extra_jsoncpp_shared
    SOURCE_DIR ${JSONCPP_SOURCE_DIR}
    PREFIX ${JSONCPP_PREFIX_DIR}/shared
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS "${JSONCPP_INSTALL_DIR}/lib/libjsoncpp.so"
    CMAKE_ARGS ${JSONCPP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS=ON
)

ExternalProject_Add(
    extra_jsoncpp_static
    SOURCE_DIR ${JSONCPP_SOURCE_DIR}
    PREFIX ${JSONCPP_PREFIX_DIR}/static
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS "${JSONCPP_INSTALL_DIR}/lib/libjsoncpp_static.a"
    CMAKE_ARGS ${JSONCPP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS=OFF
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
        COMMAND ${CMAKE_COMMAND} -E rename
            ${JSONCPP_INSTALL_DIR}/lib/libjsoncpp.a
            ${JSONCPP_INSTALL_DIR}/lib/libjsoncpp_static.a
    DEPENDS extra_jsoncpp_shared
)

add_custom_target(extra_jsoncpp DEPENDS extra_jsoncpp_shared extra_jsoncpp_static)

set(JSONCPP_ROOT_DIR ${JSONCPP_INSTALL_DIR})
set(JSONCPP_DIR ${JSONCPP_INSTALL_DIR})

if(NOT EXISTS ${JSONCPP_INCLUDE_DIR})
    file(MAKE_DIRECTORY ${JSONCPP_INCLUDE_DIR})
endif()

add_library(jsoncpp_includes INTERFACE IMPORTED GLOBAL)
set_target_properties(jsoncpp_includes PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIR}"
)
add_dependencies(jsoncpp_includes extra_jsoncpp)

add_library(jsoncpp_lib SHARED IMPORTED GLOBAL)
set_target_properties(jsoncpp_lib PROPERTIES
    IMPORTED_LOCATION "${JSONCPP_INSTALL_DIR}/lib/libjsoncpp.so"
    INTERFACE_LINK_LIBRARIES "jsoncpp_includes"
)

add_library(jsoncpp_static STATIC IMPORTED GLOBAL)
set_target_properties(jsoncpp_static PROPERTIES
    IMPORTED_LOCATION "${JSONCPP_INSTALL_DIR}/lib/libjsoncpp_static.a"
    INTERFACE_LINK_LIBRARIES "jsoncpp_includes"
)

if(DEFINED BUILD_TESTING_OLD)
    set(BUILD_TESTING ${BUILD_TESTING_OLD})
endif()

if(DEFINED BUILD_SHARED_LIBS_OLD)
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_OLD})
endif()

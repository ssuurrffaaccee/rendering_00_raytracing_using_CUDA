cmake_minimum_required(VERSION 3.26)
project(
  cuda_by_example
  VERSION 1.0.0
  LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(glm)
find_package(stb)
find_package(glfw3)
find_package(glad)
find_package(assimp)


set(LINK_TARGET glm::glm stb::stb glfw glad::glad assimp::assimp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
endif()
include_directories("${PROJECT_SOURCE_DIR}/src")


function(learn_step step_name)  
    file(GLOB_RECURSE SRC_FILES_CPP CONFIGURE_DEPENDS "src/${step_name}/*.cpp")
    file(GLOB_RECURSE SRC_FILES_CU CONFIGURE_DEPENDS "src/${step_name}/*.cu")
    set(SRC_FILES ${SRC_FILES_CPP} ${SRC_FILES_CU})
    add_executable(${step_name} ${SRC_FILES})
    target_link_libraries(${step_name} ${LINK_TARGET})
    message("${PROJECT_SOURCE_DIR}/src/${step_name}/")
    target_include_directories(${step_name} PRIVATE "${PROJECT_SOURCE_DIR}/src/${step_name}/")
    install(TARGETS ${step_name} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin/${step_name})
    file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS "src/${step_name}/*.frag" "src/${step_name}/*.vert")
    foreach(shader_path ${SHADER_FILES})
      install(
        FILES ${shader_path}
        DESTINATION ${PROJECT_SOURCE_DIR}/bin/${step_name}
      )
    endforeach()
endfunction()

learn_step(x00-viewer_cpu)
learn_step(x01-viewer_gpu)
learn_step(cu00-enum_gpu)
learn_step(cu01-set_gpu)
learn_step(cu02-simple_kennal_param)
learn_step(cu03-loop_gpu)
learn_step(cu04-loop_grid_stride_gpu)
learn_step(cu05-julia_gpu)
learn_step(cu06-add_loop_blocks)
learn_step(cu07-add_loop_blocks_grid_stride)
learn_step(cu08-dot)
learn_step(cu09-ripple)
learn_step(cu10-ripple_shared_mem)
learn_step(cu11-ray)
learn_step(cu12-ray_const)
learn_step(cu13-heat)

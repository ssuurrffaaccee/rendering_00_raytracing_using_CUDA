cmake_minimum_required(VERSION 3.26)
project(
  learning_ecs
  VERSION 1.0.0
  LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

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

function(move_shader step_name shader_paths)
   foreach(shader_path ${shader_paths})
     install(
       FILES ${shader_path}
       DESTINATION ${PROJECT_SOURCE_DIR}/bin/${step_name}
     )
  endforeach()
endfunction()

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

learn_step(cuda_raytracing_in_one_weekend)
learn_step(cuda_raytracing_the_next_week)



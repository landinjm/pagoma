#
# CMake script
#

set(TARGET "pagoma")

set(TARGET_SRC ${TARGET}.cc)

cmake_minimum_required(VERSION 3.13.4)

find_package(
  deal.II
  9.7.0
  HINTS
    ${deal.II_DIR}
    ${DEAL_II_DIR}
    $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not find a valid deal.II installation.")
endif()

if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_LAPACK)
  message(FATAL_ERROR "Could not find a valid deal.II installation.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()

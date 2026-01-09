#pragma once

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/data_out.h>

#include <type_traits>

namespace utitilies {

template<typename number, typename memory_space, typename memory_space_2>
void
transfer_solution_between_memory_spaces(
  const dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec,
  dealii::LinearAlgebra::distributed::Vector<number, memory_space_2>& vec2,
  const dealii::IndexSet& locally_owned_dofs)
{

  Assert(!(std::is_same_v<memory_space, memory_space_2>),
         dealii::ExcMessage("To transfer a vector between memory spaces, "
                            "the two template parameters must belong to "
                            "different memory spaces."));

  dealii::LinearAlgebra::ReadWriteVector<number> rw_vector(locally_owned_dofs);
  rw_vector.import_elements(vec, dealii::VectorOperation::insert);
  vec2.import_elements(rw_vector, dealii::VectorOperation::insert);
}

template<unsigned int dim, typename number, typename memory_space>
void
dump_vector_to_vtu(
  const dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec,
  const dealii::DoFHandler<dim>& dof_handler,
  const std::string& filename)
{

  // TODO: Add checks that the dof handler and vector are the same size.

  // Grab the index range for the local size of the vector
  const dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

  // Create a new vector that belongs on the host memory. We will then copy data
  // to this. It's not the most efficient, but this function should only be
  // called when debugging.
  dealii::LinearAlgebra::distributed::Vector<number, dealii::MemorySpace::Host>
    host_vector(locally_owned_dofs, vec.get_mpi_communicator());

  if constexpr (std::is_same_v<memory_space, dealii::MemorySpace::Default>) {
    transfer_solution_between_memory_spaces(
      vec, host_vector, locally_owned_dofs);
  } else if constexpr (std::is_same_v<memory_space,
                                      dealii::MemorySpace::Host>) {
    host_vector = vec;
  }

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(host_vector, "solution");
  data_out.build_patches();
  dealii::DataOutBase::VtkFlags flags;
  flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_in_parallel(filename + ".vtu", vec.get_mpi_communicator());
}

} // namespace utitilies

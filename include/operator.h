#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace pagoma {

// Parameters for the mobility and gradient energy
constexpr double mobility = 1.0;
constexpr double gradient_energy = 2.0;

namespace GPU {

/**
 * @brief User-defined operator evaluated at each quadrature point.
 */
template<unsigned int dim,
         unsigned int degree,
         typename number,
         unsigned int index>
class UserOperator
{
public:
  DEAL_II_HOST_DEVICE UserOperator() {};

  DEAL_II_HOST_DEVICE void operator()(
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, number>* fe_eval,
    const unsigned int q_point) const
  {
    if constexpr (index == 0) {
      auto value = fe_eval->get_value(q_point);
      auto gradient = fe_eval->get_gradient(q_point);

      auto double_well = 4.0 * value * (value - 1.0) * (value - 0.5);
      auto value_submission = -mobility * double_well;
      auto gradient_submission = -mobility * gradient_energy * gradient;

      fe_eval->submit_value(value_submission, q_point);
      fe_eval->submit_gradient(gradient_submission, q_point);
    }
  }

  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs =
    dealii::Utilities::pow(degree + 1, dim);
};

template<unsigned int dim, unsigned int degree, typename number>
class LocalOperator
{
public:
  DEAL_II_HOST_DEVICE LocalOperator() {};

  DEAL_II_HOST_DEVICE void operator()(
    const typename dealii::Portable::MatrixFree<dim, number>::Data* data,
    const dealii::Portable::DeviceVector<number>& src,
    dealii::Portable::DeviceVector<number>& dst) const
  {
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval(
      data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(dealii::EvaluationFlags::EvaluationFlags::values |
                     dealii::EvaluationFlags::EvaluationFlags::gradients);
    fe_eval.apply_for_each_quad_point(UserOperator<dim, degree, number, 0>());
    fe_eval.integrate(dealii::EvaluationFlags::EvaluationFlags::values |
                      dealii::EvaluationFlags::EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  };

  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs =
    dealii::Utilities::pow(degree + 1, dim);
};

/**
 * @brief Main operator class that handles vector initialization, matrix-free
 * data, and vector multiplication.
 */
template<unsigned int dim, unsigned int degree, typename number>
class Operator : public dealii::EnableObserverPointer
{
public:
  Operator(const dealii::DoFHandler<dim, dim>& dof_handler,
           const dealii::AffineConstraints<number>& constraints)
  {
    // TODO: Mapping and quadrature should be constructor inputs
    const dealii::MappingQ<dim> mapping(degree);
    const dealii::QGaussLobatto<1> quadrature(degree + 1);
    typename dealii::Portable::MatrixFree<dim, number>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = dealii::update_values |
                                           dealii::update_gradients |
                                           dealii::update_JxW_values;
    data.reinit(mapping, dof_handler, constraints, quadrature, additional_data);
  }

  template<typename vector>
  void vmult(vector& dst, const vector& src) const
  {
    dst = 0.0;
    data.cell_loop(LocalOperator<dim, degree, number>(), src, dst);
    // data.copy_constrained_values(src, dst);
  }

  template<typename memory_space>
  void initialize_dof_vector(
    dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec) const
  {
    data.initialize_dof_vector(vec);
  };

  dealii::Portable::MatrixFree<dim, number>* get_matrix_free_data()
  {
    return &data;
  }

private:
  dealii::Portable::MatrixFree<dim, number> data;
};

}

namespace CPU {

template<unsigned int dim, unsigned int degree, typename number>
class LocalOperator
{
public:
  LocalOperator() {};

  template<typename VectorType>
  void operator()(const dealii::MatrixFree<dim, number>& data,
                  VectorType& dst,
                  const VectorType& src,
                  const std::pair<unsigned int, unsigned int>& cell_range) const
  {
    dealii::FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(dealii::EvaluationFlags::values |
                       dealii::EvaluationFlags::gradients);

      for (unsigned int q_point = 0; q_point < fe_eval.n_q_points; ++q_point) {
        auto value = fe_eval.get_value(q_point);
        auto gradient = fe_eval.get_gradient(q_point);

        auto double_well = 4.0 * value * (value - 1.0) * (value - 0.5);
        auto value_submission = -mobility * double_well;
        auto gradient_submission = -mobility * gradient_energy * gradient;

        fe_eval.submit_value(value_submission, q_point);
        fe_eval.submit_gradient(gradient_submission, q_point);
      }

      fe_eval.integrate(dealii::EvaluationFlags::values |
                        dealii::EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }
  }
};

/**
 * @brief Main operator class that handles vector initialization, matrix-free
 * data, and vector multiplication.
 */
template<unsigned int dim, unsigned int degree, typename number>
class Operator : public dealii::EnableObserverPointer
{
public:
  Operator(const dealii::DoFHandler<dim, dim>& dof_handler,
           const dealii::AffineConstraints<number>& constraints)
    : local_operator()
  {
    // TODO: Mapping and quadrature should be constructor inputs
    const dealii::MappingQ<dim> mapping(degree);
    const dealii::QGaussLobatto<1> quadrature(degree + 1);
    typename dealii::MatrixFree<dim, number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = dealii::update_values |
                                           dealii::update_gradients |
                                           dealii::update_JxW_values;
    data.reinit(mapping, dof_handler, constraints, quadrature, additional_data);
  }

  template<typename vector>
  void vmult(vector& dst, const vector& src) const
  {
    data.cell_loop(
      &LocalOperator<dim, degree, number>::template operator()<vector>,
      &local_operator,
      dst,
      src,
      true);
  }

  template<typename memory_space>
  void initialize_dof_vector(
    dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec) const
  {
    data.initialize_dof_vector(vec);
  };

  dealii::MatrixFree<dim, number>* get_matrix_free_data() { return &data; }

private:
  dealii::MatrixFree<dim, number> data;
  LocalOperator<dim, degree, number> local_operator;
};

}
}

#pragma once

#include <deal.II/base/exceptions.h>

namespace pagoma {

template<typename MatrixType>
class TimeIterator
{
public:
  TimeIterator(const MatrixType& _matrix)
    : matrix(_matrix) {};

  /**
   * @brief Explicit update with no time component.
   *
   * This is simply the a vmult scaled by the invm.
   */
  template<typename VectorType>
  void no_update_explicit(VectorType& dst,
                          const VectorType& src,
                          const VectorType& invm) const
  {
    matrix.vmult(dst, src);
    dst.scale(invm);
  }

  /**
   * @brief Forward euler update.
   */
  template<typename VectorType>
  void forward_euler(VectorType& dst,
                     const VectorType& src,
                     VectorType& rate,
                     const VectorType& invm,
                     typename VectorType::value_type timestep) const
  {
    dst = src;
    matrix.vmult(rate, src);
    rate.scale(invm);
    dst.add(timestep, rate);
  }

  /**
   * @brief Forward euler update.
   *
   * Rather than explicitly pass a vector to hold the rate calculation, this
   * constructs one on the fly using the same parallel layout as the src vector.
   *
   * This will always be slower since we have to allocate memory on the fly. The
   * only real usage is to not have to deal with allocating vectors, which is
   * particularly messy with higher order schemes like RK4.
   */
  template<typename VectorType>
  void forward_euler(VectorType& dst,
                     const VectorType& src,
                     const VectorType& invm,
                     typename VectorType::value_type timestep) const
  {
    dst = src;
    VectorType rate(src);
    matrix.vmult(rate, src);
    rate.scale(invm);
    dst.add(timestep, rate);
  }

  template<typename VectorType>
  void rk4(VectorType& dst,
           const VectorType& src,
           VectorType& k1,
           VectorType& k2,
           VectorType& k3,
           VectorType& k4,
           VectorType& temp,
           const VectorType& invm,
           typename VectorType::value_type timestep) const
  {
    matrix.vmult(k1, src);
    k1.scale(invm);

    temp = src;
    temp.add(0.5 * timestep, k1);
    matrix.vmult(k2, temp);
    k2.scale(invm);

    temp = src;
    temp.add(0.5 * timestep, k2);
    matrix.vmult(k3, temp);
    k3.scale(invm);

    temp = src;
    temp.add(timestep, k3);
    matrix.vmult(k4, temp);
    k4.scale(invm);

    dst = src;
    dst.add(timestep / 6.0, k1);
    dst.add(timestep / 3.0, k2);
    dst.add(timestep / 3.0, k3);
    dst.add(timestep / 6.0, k4);
  }

  template<typename VectorType,
           typename SolverType,
           typename PreconditionerType>
  void no_update_implicit(VectorType& dst,
                          const VectorType& src,
                          SolverType& solver,
                          const PreconditionerType& preconditioner) const
  {
    solver.solve(matrix, dst, src, preconditioner);
  }

  template<typename VectorType,
           typename SolverType,
           typename PreconditionerType>
  void backward_euler(VectorType& dst,
                      const VectorType& src,
                      VectorType& rate,
                      SolverType& solver,
                      const PreconditionerType& preconditioner,
                      typename VectorType::value_type timestep) const
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented"));
  }

private:
  const MatrixType& matrix;
};

}

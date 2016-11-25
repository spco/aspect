//
//  vector_dependent_function.cpp
//  step-32
//
//  Created by Samuel Cox on 25/02/2015.
//  Copyright (c) 2015-2016 Samuel Cox. All rights reserved.
//

#include "coefficient_function.h"

// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <vector>

using namespace dealii;

template <int dim, typename Number>
const unsigned int CoefficientFunction<dim, Number>::dimension;
template <int dim, typename Number>
CoefficientFunction<dim, Number>::CoefficientFunction (const unsigned int n_components,
                                                       const Number initial_time)
  :
//CoefficientFunctionTime<Number>(initial_time),
  n_components(n_components),
  time(initial_time)
{
  // avoid the construction of function
  // objects that don't return any
  // values. This doesn't make much sense in
  // the first place, but will lead to odd
  // errors later on (happened to me in fact
  // :-)
  Assert (n_components > 0,
          ExcZero());
}
template <int dim, typename Number>
CoefficientFunction<dim, Number>::~CoefficientFunction ()
{}
template <int dim, typename Number>
CoefficientFunction<dim, Number> &CoefficientFunction<dim, Number>::operator= (const CoefficientFunction &f)
{
  AssertDimension (n_components, f.n_components);
  return *this;
}
template <int dim, typename Number>
Number CoefficientFunction<dim, Number>::value (const Point<dim> &,
                                                const Number &,
                                                const unsigned int) const
{
  Assert (false, ExcPureFunctionCalled());
  return 0;
}
template <int dim, typename Number>
Number CoefficientFunction<dim, Number>::value (const Point<dim> &,
                                                const Number &,
                                                const Tensor<1,dim,Number> &,
                                                const Tensor<1,dim,Number> &,
                                                const unsigned int) const
{
  Assert (false, ExcPureFunctionCalled());
  return 0;
}
template <int dim, typename Number>
Number CoefficientFunction<dim, Number>::value (const Point<dim> &,
                                                const Number &,
                                                const Number &,
                                                const Tensor<1,dim,Number> &,
                                                const Tensor<1,dim,Number> &,
                                                const unsigned int) const
{
  Assert (false, ExcPureFunctionCalled());
  return 0;
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_value (const Point<dim> &p,
                                                     const Number &temperature,
                                                     Vector<Number> &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    v(i) = value(p, temperature, i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_value (const Point<dim> &p,
                                                     const Number &temperature,
                                                     const Tensor<1,dim,Number> &tensor_1,
                                                     const Tensor<1,dim,Number> &tensor_2,
                                                     Vector<Number> &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    v(i) = value(p, temperature, tensor_1, tensor_2, i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_value (const Point<dim> &p,
                                                     const Number &temperature,
                                                     const Number &temperature2,
                                                     const Tensor<1,dim,Number> &tensor_1,
                                                     const Tensor<1,dim,Number> &tensor_2,
                                                     Vector<Number> &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    v(i) = value(p, temperature, temperature2, tensor_1, tensor_2, i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::value_list (const std::vector<Point<dim> > &points,
                                                   const std::vector<Number> &temperatures,
                                                   std::vector<Number> &values,
                                                   const unsigned int component) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    values[i] = this->value (points[i], temperatures[i], component);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::value_list (const std::vector<Point<dim> > &points,
                                                   const std::vector<Number> &temperatures,
                                                   const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                                   const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                                   std::vector<Number> &values,
                                                   const unsigned int component) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  Assert (tensor_1.size() == points.size(),
          ExcDimensionMismatch(tensor_1.size(), points.size()));
  Assert (tensor_2.size() == points.size(),
          ExcDimensionMismatch(tensor_2.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    values[i] = this->value (points[i], temperatures[i], tensor_1[i], tensor_2[i], component);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::value_list (const std::vector<Point<dim> > &points,
                                                   const std::vector<Number> &temperatures,
                                                   const std::vector<Number> &temperatures2,
                                                   const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                                   const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                                   std::vector<Number> &values,
                                                   const unsigned int component) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  Assert (temperatures2.size() == points.size(),
          ExcDimensionMismatch(temperatures2.size(), points.size()));
  Assert (tensor_1.size() == points.size(),
          ExcDimensionMismatch(tensor_1.size(), points.size()));
  Assert (tensor_2.size() == points.size(),
          ExcDimensionMismatch(tensor_2.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    values[i] = this->value (points[i], temperatures[i], temperatures2[i], tensor_1[i], tensor_2[i], component);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_value_list (const std::vector<Point<dim> > &points,
                                                          const std::vector<Number> &temperatures,
                                                          std::vector<Vector<Number> > &values) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    this->vector_value (points[i], temperatures[i], values[i]);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_value_list (const std::vector<Point<dim> > &points,
                                                          const std::vector<Number> &temperatures,
                                                          const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                                          const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                                          std::vector<Vector<Number> > &values) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    this->vector_value (points[i], temperatures[i], tensor_1[i], tensor_2[i], values[i]);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_value_list (const std::vector<Point<dim> > &points,
                                                          const std::vector<Number> &temperatures,
                                                          const std::vector<Number> &temperatures2,
                                                          const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                                          const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                                          std::vector<Vector<Number> > &values) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  Assert (temperatures2.size() == points.size(),
          ExcDimensionMismatch(temperatures2.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    this->vector_value (points[i], temperatures[i], temperatures2[i], tensor_1[i], tensor_2[i], values[i]);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_values (
  const std::vector<Point<dim> > &points,
  const std::vector<Number> &temperatures,
  std::vector<std::vector<Number> > &values) const
{
  const unsigned int n = this->n_components;
  AssertDimension (values.size(), n);
  AssertDimension (temperatures.size(), n);
  for (unsigned int i=0; i<n; ++i)
    value_list(points, temperatures, values[i], i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_values (
  const std::vector<Point<dim> > &points,
  const std::vector<Number> &temperatures,
  const std::vector<Tensor<1,dim,Number> > &tensor_1,
  const std::vector<Tensor<1,dim,Number> > &tensor_2,
  std::vector<std::vector<Number> > &values) const
{
  const unsigned int n = this->n_components;
  AssertDimension (values.size(), n);
  AssertDimension (temperatures.size(), n);
  for (unsigned int i=0; i<n; ++i)
    value_list(points, temperatures, tensor_1, tensor_2, values[i], i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_values (
  const std::vector<Point<dim> > &points,
  const std::vector<Number> &temperatures,
  const std::vector<Number> &temperatures2,
  const std::vector<Tensor<1,dim,Number> > &tensor_1,
  const std::vector<Tensor<1,dim,Number> > &tensor_2,
  std::vector<std::vector<Number> > &values) const
{
  const unsigned int n = this->n_components;
  AssertDimension (values.size(), n);
  AssertDimension (temperatures.size(), n);
  for (unsigned int i=0; i<n; ++i)
    value_list(points, temperatures, temperatures2, tensor_1, tensor_2, values[i], i);
}
template <int dim, typename Number>
Tensor<1,dim,Number> CoefficientFunction<dim, Number>::gradient (const Point<dim> &,
                                                                 const Number &,
                                                                 const unsigned int) const
{
  Assert (false, ExcPureFunctionCalled());
  return Tensor<1,dim,Number>();
}
template <int dim, typename Number>
Tensor<1,dim,Number> CoefficientFunction<dim, Number>::gradient (const Point<dim> &,
                                                                 const Number &,
                                                                 const Tensor<1,dim,Number> &,
                                                                 const unsigned int) const
{
  Assert (false, ExcPureFunctionCalled());
  return Tensor<1,dim,Number>();
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_gradient (const Point<dim> &p,
                                                        const Number &temperature,
                                                        std::vector<Tensor<1,dim,Number> > &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    v[i] = gradient(p, temperature, i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_gradient (const Point<dim> &p,
                                                        const Number &temperature,
                                                        const Tensor<1,dim,Number> &temperature_gradient,
                                                        std::vector<Tensor<1,dim,Number> > &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    v[i] = gradient(p, temperature, temperature_gradient, i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::gradient_list (const std::vector<Point<dim> > &points,
                                                      const std::vector<Number> &temperatures,
                                                      std::vector<Tensor<1,dim,Number> > &gradients,
                                                      const unsigned int component) const
{
  Assert (gradients.size() == points.size(),
          ExcDimensionMismatch(gradients.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    gradients[i] = gradient(points[i], temperatures[i], component);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::gradient_list (const std::vector<Point<dim> > &points,
                                                      const std::vector<Number> &temperatures,
                                                      const std::vector<Tensor<1,dim,Number> > &temperature_gradients,
                                                      std::vector<Tensor<1,dim,Number> > &gradients,
                                                      const unsigned int component) const
{
  Assert (gradients.size() == points.size(),
          ExcDimensionMismatch(gradients.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  Assert (temperature_gradients.size() == points.size(),
          ExcDimensionMismatch(temperature_gradients.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    gradients[i] = gradient(points[i], temperatures[i], temperature_gradients[i], component);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_gradients (const std::vector<Point<dim> > &p,
                                                         const std::vector<Number> &temperature,
                                                         std::vector<std::vector<Tensor<1,dim,Number> > > &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    gradient_list(p, temperature, v[i], i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_gradients (const std::vector<Point<dim> > &p,
                                                         const std::vector<Number> &temperature,
                                                         const std::vector<Tensor<1,dim,Number> > &temperature_gradient,
                                                         std::vector<std::vector<Tensor<1,dim,Number> > > &v) const
{
  AssertDimension(v.size(), this->n_components);
  for (unsigned int i=0; i<this->n_components; ++i)
    gradient_list(p, temperature, temperature_gradient, v[i], i);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_gradient_list (const std::vector<Point<dim> > &points,
                                                             const std::vector<Number> &temperatures,
                                                             std::vector<std::vector<Tensor<1,dim,Number> > > &gradients) const
{
  Assert (gradients.size() == points.size(),
          ExcDimensionMismatch(gradients.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    {
      Assert (gradients[i].size() == n_components,
              ExcDimensionMismatch(gradients[i].size(), n_components));
      vector_gradient (points[i], temperatures[i], gradients[i]);
    }
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_gradient_list (const std::vector<Point<dim> > &points,
                                                             const std::vector<Number> &temperatures,
                                                             const std::vector<Tensor<1,dim,Number> > &temperature_gradients,
                                                             std::vector<std::vector<Tensor<1,dim,Number> > > &gradients) const
{
  Assert (gradients.size() == points.size(),
          ExcDimensionMismatch(gradients.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  Assert (temperature_gradients.size() == points.size(),
          ExcDimensionMismatch(temperature_gradients.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    {
      Assert (gradients[i].size() == n_components,
              ExcDimensionMismatch(gradients[i].size(), n_components));
      vector_gradient (points[i], temperatures[i], temperature_gradients[i], gradients[i]);
    }
}
template <int dim, typename Number>
Number CoefficientFunction<dim, Number>::laplacian (const Point<dim> &,
                                                    const Number &,
                                                    const unsigned int) const
{
  Assert (false, ExcPureFunctionCalled());
  return 0;
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_laplacian (const Point<dim> &,
                                                         const Number &,
                                                         Vector<Number> &) const
{
  Assert (false, ExcPureFunctionCalled());
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::laplacian_list (const std::vector<Point<dim> > &points,
                                                       const std::vector<Number> &temperatures,
                                                       std::vector<Number> &laplacians,
                                                       const unsigned int component) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (laplacians.size() == points.size(),
          ExcDimensionMismatch(laplacians.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    laplacians[i] = this->laplacian (points[i], temperatures[i], component);
}
template <int dim, typename Number>
void CoefficientFunction<dim, Number>::vector_laplacian_list (const std::vector<Point<dim> > &points,
                                                              const std::vector<Number> &temperatures,
                                                              std::vector<Vector<Number> > &laplacians) const
{
  // check whether component is in
  // the valid range is up to the
  // derived class
  Assert (laplacians.size() == points.size(),
          ExcDimensionMismatch(laplacians.size(), points.size()));
  Assert (temperatures.size() == points.size(),
          ExcDimensionMismatch(temperatures.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    this->vector_laplacian (points[i], temperatures[i], laplacians[i]);
}
template <int dim, typename Number>
std::size_t
CoefficientFunction<dim, Number>::memory_consumption () const
{
  // only simple data elements, so
  // use sizeof operator
  return sizeof (*this);
}

template<int dim, typename Number>
void
CoefficientFunction<dim, Number>::set_time (const Number new_time)
{
  time = new_time;
}
template<int dim, typename Number>
void
CoefficientFunction<dim, Number>::advance_time (const Number delta_t)
{
  set_time (time+delta_t);
}

namespace dealii
{
  template class CoefficientFunction<1, double>;
  template class CoefficientFunction<2, double>;
  template class CoefficientFunction<3, double>;
}


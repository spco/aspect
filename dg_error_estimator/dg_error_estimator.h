/*
 Copyright (C) 2016 by Sam Cox, University of Leicester.

 This file is a custom plugin to ASPECT, and thus is not a part of ASPECT for general release.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file doc/COPYING.  If not see
 <http://www.gnu.org/licenses/>.
 */


#ifndef __aspect__mesh_refinement_dg_error_estimator_h
#define __aspect__mesh_refinement_dg_error_estimator_h

#include <aspect/mesh_refinement/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include "coefficient_function.h"

namespace dealii
{
  namespace EquationData
  {
    const double alpha = 1.;

    template <int dim, typename Number>
    class WeightingFunction : public CoefficientFunction<dim, Number>
    {
      public:
        WeightingFunction () : CoefficientFunction<dim, Number>(1) {}

        double value (const Point<dim>   &p,
                      const Number &helmholtz,
                      const unsigned int component = 0) const;
        Tensor<1,dim,Number> gradient (const Point<dim>  &p,
                                       const Number &helmholtz,
                                       const Tensor<1,dim,Number> &helmholtz_gradient,
                                       const unsigned int component = 0) const;
    };

    template <int dim, typename Number>
    double
    WeightingFunction<dim, Number>::value (const Point<dim>  &p,
                                           const Number &helmholtz,
                                           const unsigned int component) const
    {
      Assert (component == 0 ,
              ExcIndexRange (component, 0, 1));
      return std::exp(-alpha*helmholtz);
    }

    template <int dim, typename Number>
    Tensor<1,dim,Number>
    WeightingFunction<dim, Number>::gradient (const Point<dim>  &p,
                                              const Number &helmholtz,
                                              const Tensor<1,dim,Number> &helmholtz_gradient,
                                              const unsigned int component) const
    {
      Assert (component == 0 ,
              ExcIndexRange (component, 0, 1));
      Tensor<1,dim,Number> output_tensor;
      for (unsigned int c=0; c<dim; ++c)
        output_tensor[c] = -alpha*helmholtz_gradient[c]*std::exp(-alpha*helmholtz);
      return output_tensor;
    }

    template <int dim, typename Number>
    class ReactionFunction : public CoefficientFunction<dim, Number>
    {
      public:
        ReactionFunction () : CoefficientFunction<dim, Number>(1) {}

        double value (const Point<dim>   &p,
                      const Number &convection_divergence,
                      const Number &diffusion,
                      const Tensor<1,dim,Number> &helmholtz_gradient,
                      const Tensor<1,dim,Number> &convection,
                      const unsigned int component = 0) const;
    };

    template <int dim, typename Number>
    double
    ReactionFunction<dim, Number>::value (const Point<dim>   &p,
                                          const Number &convection_divergence,
                                          const Number &diffusion,
                                          const Tensor<1,dim,Number> &helmholtz_gradient,
                                          const Tensor<1,dim,Number> &convection,
                                          const unsigned int component) const
    {
      Assert (component == 0 ,
              ExcIndexRange (component, 0, 1));
      //            return std::max(0., 0.1 + (2. + 0.) *(- 0.5 * (alpha * (helmholtz_gradient * convection)
      //                                             - (1-alpha*diffusion)*convection_divergence
      //                                             - alpha*alpha * diffusion * helmholtz_gradient.norm()
      //                                             )
      //                                       )
      //                            );
      return std::max(0.0, 0.01 + (2. + 0.) *(- 1. * (alpha * (helmholtz_gradient * convection)
                                                      - (1-alpha*diffusion)*convection_divergence
                                                      - alpha*alpha * diffusion * helmholtz_gradient * helmholtz_gradient
                                                     )
                                             )
                     );

      //            return std::max(0., 0.5*(0.4 - 2. *
      //                                     (0.5 * (alpha * (helmholtz_gradient * convection)
      //                                             - (1-alpha*diffusion)*convection_divergence
      //                                             - alpha*alpha * diffusion * helmholtz_gradient.norm()
      //                                             )
      //                                      )
      //                                     +
      //                                     std::sqrt(4*(- 0.5 * (alpha * (helmholtz_gradient * convection)
      //                                                           - (1-alpha*diffusion)*convection_divergence
      //                                                           - alpha*alpha * diffusion * helmholtz_gradient.norm()
      //                                                           )
      //                                                  )*(- 0.5 * (alpha * (helmholtz_gradient * convection)
      //                                                              - (1-alpha*diffusion)*convection_divergence
      //                                                              - alpha*alpha * diffusion * helmholtz_gradient.norm()
      //                                                              )
      //                                                     )+0.16))
      //                            );

    }

  }
}


namespace aspect
{
  namespace MeshRefinement
  {
    template <class DH>
    struct DGParallelData
    {
      static const unsigned int dim      = DH::dimension;
      static const unsigned int spacedim = DH::space_dimension;

      /**
       * The quadrature formulas to be used for the cells.
       */
      const Quadrature<dim>                           cell_quadrature;

      /**
       * The quadrature formulas to be used for the faces.
       */
      const Quadrature<dim-1>                         face_quadrature;

      /**
       * FEValues object to integrate over the cell.
       */
      FEValues<dim,spacedim>                          fe_values_cell;

      FEValues<dim,spacedim>                          fe_values_neighbor;

      FEValues<dim,spacedim>                          helmholtz_fe_values_cell;

      FEValues<dim,spacedim>                          helmholtz_fe_values_neighbor;

      /**
       * FEFaceValues objects to integrate over the faces of the current and
       * potentially of neighbor cells.
       */
      FEFaceValues<dim,spacedim>                      fe_face_values_cell;
      FEFaceValues<dim,spacedim>                      fe_face_values_neighbor;
      FESubfaceValues<dim,spacedim>                   fe_subface_values;

      /**
       * A vector to store the value of temperature and old
       * temperature in the quadrature points
       *
       * size = no of q_points
       */
      std::vector<double>                             temperature_values;

      std::vector<double>                             old_temperature_values;

      /**
       * A vector to store the rhs projection in the quadrature
       * points
       *
       * size = no of q_points
       */
      std::vector<double>                             rhs_projection_values;

      /**
       * A vector to store the gradient in the quadrature
       * points
       * size = no of q_points // Tensor of grad in each direction in space
       */
      std::vector<Tensor<1,spacedim> >                temperature_gradients;

      /**
       * A vector to store the temperature laplacian in the quadrature
       * points
       *
       * size = no of q_points
       */
      std::vector<double>                             temperature_laplacians;

      /**
       * A vector to store the jump of the normal gradients
       * ( [grad T] = (grad T - grad T_e) \cdot n ) in the quadrature
       * points
       *
       * size = no of q_points
       */
      std::vector<double>                             temperature_face_scaled_gradients_jump;

      /**
       * A vector for the gradients of the finite element function on one cell
       *
       * size = no of q_points
       */
      std::vector<Tensor<1,spacedim> >                temperature_face_gradients_cell;

      /**
       * The same vector for a neighbor cell
       */
      std::vector<Tensor<1,spacedim> >                temperature_face_gradients_neighbor;

      /**
       * A vector to store the jump of the values in the quadrature
       * points
       *
       * Note that we wish to calculate ||(T-T_e)n||^2 = ||[T]n||^2. Then note that
       * int( ((S)n)^2 )) = int( ((S)n_1)^2 + ((S)n_2)^2) = int( (S) (n_1^2 + n_2^2) ) = int( (S) 1 ).
       * So we don't need to calculate the multiplication by n, instead it's just int( (S)).
       * Now take S = T-T_e.
       *
       * size = no of q_points
       */

      std::vector<double>                             temperature_face_values_jump;

      /**
       * A vector to store the values in the quadrature
       * points
       *
       * size = no of q_points
       */
      std::vector<double>                             temperature_face_values_cell;

      /**
       * The same vector for a neighbor cell
       */
      std::vector<double>                             temperature_face_values_neighbor;

      /**
       * A vector to store the stokes value in the quadrature
       * points on the cell and face
       *
       * size = no of q_points // Vector containing u components
       */
      std::vector<Tensor<1,spacedim> >                velocity_values;

      std::vector<Tensor<1,spacedim> >                velocity_face_values_cell;

      std::vector<double>                             div_u_cell;

      std::vector<Tensor<1,spacedim> >                velocity_values_neighbor;

      std::vector<double>                             div_u_neighbor;

      /**
       * The normal vectors of the finite element function on one face
       *
       * size = no of q_points
       */

      std::vector<Tensor<1,spacedim> >                normal_vectors;

      /**
       * A vector of double to hold diffusion values
       * at each quadrature point on a cell
       *
       * size = no of q_points
       */
      std::vector<double>                             diffusion_values_cell;
      std::vector<double>                             diffusion_values_neighbor;

      /**
       * A vector of double to hold diffusion values
       * at each quadrature point on a face and neighbor's face
       *
       * size = no of q_points
       */
      std::vector<double>                             diffusion_values_face_cell;

      std::vector<double>                             diffusion_values_face_neighbor;

      std::vector<double>                             value_H;

      std::vector<Tensor<1,dim,double> >              gradient_H;

      std::vector<double>                             value_H_neighbor;

      std::vector<Tensor<1,dim,double> >              gradient_H_neighbor;

      std::vector<double>                             face_value_H;

      std::vector<Tensor<1,spacedim> >                face_gradient_H; //is this needed?

      std::vector<double>                             reaction_values;

      /**
       * Array for the products of Jacobian determinants and weights of
       * quadrature points in cells
       */
      std::vector<double>                             JxW_cell_values;

      /**
       * Array for the products of Jacobian determinants and weights of
       * quadrature points on faces.
       */
      std::vector<double>                             JxW_face_values;

      double                                          edgevertexpatch_max_conv_squared;

      double                                          timestep;

      const unsigned int                              n_cell_terms;
      const unsigned int                              n_face_terms;

      LinearAlgebra::Vector                           helmholtz_solution;

      const EquationData::WeightingFunction<dim,double>       *weighting_function;

      const EquationData::ReactionFunction<dim,double>        *reaction_function;

      /**
       * Constructor.
       */
      DGParallelData (const FiniteElement<dim>                 &finite_element,
                      const Quadrature<dim>                    &cell_quadrature,
                      const Quadrature<dim-1>                  &face_quadrature,
                      const FiniteElement<dim>                 &helmholtz_finite_element,
                      const Quadrature<dim>                    &helmholtz_cell_quadrature,
                      const Mapping<dim>                       &mapping,
                      const bool                                need_quadrature_points,
                      const double                              timestep_in,
                      unsigned int                              n_cell_terms,
                      unsigned int                              n_face_terms,
                      LinearAlgebra::Vector                    &helmholtz_solution,
                      const EquationData::WeightingFunction<dim,double> *weighting_function,
                      const EquationData::ReactionFunction<dim,double>  *reaction_function);

      /**
       * Resize the arrays so that they fit the number of quadrature points
       * associated with the given finite element index into the hp
       * collections.
       */
      void resize ();
    };


    template <class DH>
    DGParallelData<DH>::
    DGParallelData (const FiniteElement<dim>                 &finite_element,
                    const Quadrature<dim>                    &cell_quadrature,
                    const Quadrature<dim-1>                  &face_quadrature,
                    const FiniteElement<dim>                 &helmholtz_finite_element,
                    const Quadrature<dim>                    &helmholtz_cell_quadrature,
                    const Mapping<dim>                       &mapping,
                    const bool                                need_quadrature_points,
                    const double                              timestep_in,
                    unsigned int                              n_cell_terms,
                    unsigned int                              n_face_terms,
                    LinearAlgebra::Vector                    &helmholtz_solution,
                    const EquationData::WeightingFunction<dim,double> *weighting_function,
                    const EquationData::ReactionFunction<dim,double>  *reaction_function)
      :
      cell_quadrature   (cell_quadrature),
      face_quadrature   (face_quadrature),
      fe_values_cell          (mapping,
                               finite_element,
                               cell_quadrature,
                               update_values         |
                               update_gradients      |
                               update_hessians       |
                               update_JxW_values     |
                               (need_quadrature_points  ?
                                update_quadrature_points :
                                UpdateFlags()) ),
      fe_values_neighbor      (mapping,
                               finite_element,
                               cell_quadrature,
                               update_values         |
                               update_gradients      |
                               update_JxW_values     |
                               (need_quadrature_points  ?
                                update_quadrature_points :
                                UpdateFlags()) ),
      helmholtz_fe_values_cell(mapping,
                               helmholtz_finite_element,
                               helmholtz_cell_quadrature,
                               update_values         |
                               update_gradients      |
                               update_hessians       |
                               update_JxW_values     |
                               (need_quadrature_points  ?
                                update_quadrature_points :
                                UpdateFlags()) ),
      helmholtz_fe_values_neighbor  (mapping,
                                     helmholtz_finite_element,
                                     helmholtz_cell_quadrature,
                                     update_values         |
                                     update_gradients      |
                                     update_JxW_values     |
                                     (need_quadrature_points  ?
                                      update_quadrature_points :
                                      UpdateFlags()) ),
      fe_face_values_cell     (mapping,
                               finite_element,
                               face_quadrature,
                               update_values         |
                               update_gradients      |
                               update_JxW_values     |
                               (need_quadrature_points  ?
                                update_quadrature_points :
                                UpdateFlags()) |
                               update_normal_vectors),
      fe_face_values_neighbor (mapping,
                               finite_element,
                               face_quadrature,
                               update_values         |
                               update_gradients |
                               update_quadrature_points),
      fe_subface_values       (mapping,
                               finite_element,
                               face_quadrature,
                               update_values    |
                               update_gradients |
                               update_quadrature_points),
      temperature_values                        (cell_quadrature.size()),
      old_temperature_values                    (cell_quadrature.size()),
      rhs_projection_values                     (cell_quadrature.size()),
      temperature_gradients                     (cell_quadrature.size()),
      temperature_laplacians                    (cell_quadrature.size()),

      temperature_face_scaled_gradients_jump    (face_quadrature.size()),
      temperature_face_gradients_cell           (face_quadrature.size()),
      temperature_face_gradients_neighbor       (face_quadrature.size()),
      temperature_face_values_jump              (face_quadrature.size()),
      temperature_face_values_cell              (face_quadrature.size()),
      temperature_face_values_neighbor          (face_quadrature.size()),

      velocity_values                           (cell_quadrature.size()),
      velocity_face_values_cell                 (face_quadrature.size()),
      div_u_cell                                (cell_quadrature.size()),
      velocity_values_neighbor                  (cell_quadrature.size()),
      div_u_neighbor                            (cell_quadrature.size()),

      normal_vectors                            (face_quadrature.size()),

      diffusion_values_cell                     (cell_quadrature.size()),
      diffusion_values_neighbor                 (cell_quadrature.size()),
      diffusion_values_face_cell                (face_quadrature.size()),
      diffusion_values_face_neighbor            (face_quadrature.size()),

      value_H                                   (cell_quadrature.size()),
      gradient_H                                (cell_quadrature.size()),

      value_H_neighbor                          (cell_quadrature.size()),
      gradient_H_neighbor                       (cell_quadrature.size()),

      face_value_H                              (cell_quadrature.size()),
      face_gradient_H                           (cell_quadrature.size()),

      reaction_values                           (cell_quadrature.size()),

      JxW_cell_values                           (cell_quadrature.size()),
      JxW_face_values                           (face_quadrature.size()),

      edgevertexpatch_max_conv_squared  (0.),
      timestep          (timestep_in),
      n_cell_terms      (n_cell_terms),
      n_face_terms      (n_face_terms),
      helmholtz_solution(helmholtz_solution),
      weighting_function(weighting_function),
      reaction_function (reaction_function)
    {}



    template <class DH>
    void
    DGParallelData<DH>::resize ()
    {
      const unsigned int n_face_q_points  = face_quadrature.size();
      const unsigned int n_cell_q_points  = cell_quadrature.size();

      temperature_values.clear();
      temperature_values.resize(n_cell_q_points);
      old_temperature_values.clear();
      old_temperature_values.resize(n_cell_q_points);
      rhs_projection_values.clear();
      rhs_projection_values.resize(n_cell_q_points);
      temperature_gradients.clear();
      temperature_gradients.resize(n_cell_q_points);
      temperature_laplacians.clear();
      temperature_laplacians.resize(n_cell_q_points);

      temperature_face_scaled_gradients_jump.clear();
      temperature_face_scaled_gradients_jump.resize(n_face_q_points);
      temperature_face_gradients_cell.clear();
      temperature_face_gradients_cell.resize(n_face_q_points);
      temperature_face_gradients_neighbor.clear();
      temperature_face_gradients_neighbor.resize(n_face_q_points);
      temperature_face_values_jump.clear();
      temperature_face_values_jump.resize(n_face_q_points);
      temperature_face_values_cell.clear();
      temperature_face_values_cell.resize(n_face_q_points);
      temperature_face_values_neighbor.clear();
      temperature_face_values_neighbor.resize(n_face_q_points);

      velocity_values.clear();
      velocity_values.resize(n_cell_q_points);
      velocity_face_values_cell.clear();
      velocity_face_values_cell.resize(n_face_q_points);
      div_u_cell.clear();
      div_u_cell.resize(n_cell_q_points);

      velocity_values_neighbor.clear();
      velocity_values_neighbor.resize(n_cell_q_points);
      div_u_neighbor.clear();
      div_u_neighbor.resize(n_cell_q_points);

      normal_vectors.resize(n_face_q_points);

      diffusion_values_cell.resize(n_cell_q_points);
      diffusion_values_face_cell.resize(n_face_q_points);
      diffusion_values_face_neighbor.resize(n_face_q_points);

      value_H.clear();
      value_H.resize(n_cell_q_points);
      gradient_H.clear();
      gradient_H.resize(n_cell_q_points);

      value_H_neighbor.clear();
      value_H_neighbor.resize(n_cell_q_points);
      gradient_H_neighbor.clear();
      gradient_H_neighbor.resize(n_cell_q_points);

      face_value_H.clear();
      face_value_H.resize(n_face_q_points);
      face_gradient_H.clear();
      face_gradient_H.resize(n_face_q_points);

      reaction_values.clear();
      reaction_values.resize(n_cell_q_points);

      JxW_cell_values.resize(n_cell_q_points);
      JxW_face_values.resize(n_face_q_points);
    }

    /**
     * A class that implements a mesh refinement criterion based on an estimator
     * derived from the zero-reaction convection diffusion problem
     *
     * @ingroup MeshRefinement
     */
    template <int dim>
    class DGErrorEstimator : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      private:
        std::vector<float>
        integrate_over_face(DGParallelData<DoFHandler<dim> >              &parallel_data,
                            const typename DoFHandler<dim>::face_iterator &face,
                            const typename DoFHandler<dim>::active_cell_iterator       &cell,
                            const typename DoFHandler<dim>::active_cell_iterator       &neighbor,
                            const typename DoFHandler<dim>::active_cell_iterator       &helmholtz_cell,
                            const typename DoFHandler<dim>::active_cell_iterator       &helmholtz_neighbor,
                            FEFaceValues<dim>                             &fe_face_values_cell
                           ) const;
        void
        integrate_over_regular_face (std::map<typename DoFHandler<dim>::face_iterator,std::vector<float> > &local_face_integrals,
                                     const typename DoFHandler<dim>::active_cell_iterator                  &cell,
                                     const unsigned int                                                     face_no,
                                     const typename DoFHandler<dim>::active_cell_iterator                  &helmholtz_cell,
                                     DGParallelData<DoFHandler<dim> >                                      &parallel_data,
                                     FEFaceValues<dim>                                                     &fe_face_values_cell,
                                     FEFaceValues<dim>                                                     &fe_face_values_neighbor) const;
        void
        integrate_over_irregular_face (std::map<typename DoFHandler<dim>::face_iterator,std::vector<float> > &local_face_integrals,
                                       const typename DoFHandler<dim>::active_cell_iterator                  &cell,
                                       const unsigned int                                                     face_no,
                                       const typename DoFHandler<dim>::active_cell_iterator                  &helmholtz_cell,
                                       DGParallelData<DoFHandler<dim> >                                      &parallel_data,
                                       FEFaceValues<dim>                                                     &fe_face_values,
                                       FESubfaceValues<dim>                                                  &fe_subface_values) const;

        void
        integrate_over_cell(std::map<typename DoFHandler<dim>::active_cell_iterator,std::vector<float> > &local_cell_integrals,
                            typename DoFHandler<dim>::active_cell_iterator                               &cell,
                            DGParallelData<DoFHandler<dim> >                                             &parallel_data,
                            typename DoFHandler<dim>::active_cell_iterator                               &projection_cell,
                            FEValues<dim>                                                                &projection_fe_values,
                            Quadrature<dim>                                                              &projection_quadrature,
                            FEValues<dim>                                                                &helmholtz_fe_values,
                            typename DoFHandler<dim>::active_cell_iterator                               &helmholtz_cell) const;

        void
        compute_helmholtz_decomposition(const FE_Q<dim> &helmholtz_fe,
                                        const DoFHandler<dim> &helmholtz_dof_handler,
                                        const UpdateFlags &helmholtz_update_flags,
                                        const QGauss<dim> &helmholtz_quadrature,
                                        FEValues<dim> &helmholtz_fe_values,
                                        LinearAlgebra::Vector &helmholtz_solution) const;
      public:
        /**
         * Execute this mesh refinement criterion.
         *
         * @param[out] error_indicators A vector that for every active cell of
         * the current mesh (which may be a partition of a distributed mesh)
         * provides an error indicator. This vector will already have the
         * correct size when the function is called.
         */
        virtual
        void
        execute (Vector<float> &error_indicators) const;
    };
  }
}

#endif

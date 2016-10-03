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


#include <aspect/mesh_refinement/dg_error_estimator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/function_lib.h>


/* Assumptions made in this estimator:
 Diffusion is constant and non-zero (this is assumed in the weightings, but not necessarily in the calculation of some terms).
 */

/* The tag FUTURE means this is what would need to be used to get the full cell-vertex-patch values for convection
 */
namespace aspect
{
  namespace MeshRefinement
  {
    template <class DH>
    struct SampleLocalIntegrals
    {
      /* local_face_integrals is a mapping from the faces of the mesh to a std::vector of floats.
       * Each element of the vector contains one of the edge terms at the face.
       */
      std::map<typename DH::face_iterator,std::vector<float> > local_face_integrals;

      /* local_cell_integrals is a mapping from the cells of the mesh to a std::vector of floats.
       * Each element of the vector contains one of the cell terms on the cell.
       */
      std::map<typename DH::active_cell_iterator,std::vector<float> > local_cell_integrals;

      //constructor
      SampleLocalIntegrals ();

      //copy constructor
      SampleLocalIntegrals (const SampleLocalIntegrals<DH> &sample_local_integrals);
    };

    template <class DH>
    SampleLocalIntegrals<DH>::SampleLocalIntegrals()
      :
      local_face_integrals(std::map<typename DH::face_iterator,       std::vector<float> >() ),
      local_cell_integrals(std::map<typename DH::active_cell_iterator,std::vector<float> >() )
    {}

    template <class DH>
    SampleLocalIntegrals<DH>::
    SampleLocalIntegrals (const SampleLocalIntegrals<DH> &sample_local_integrals)
      :
      local_face_integrals (sample_local_integrals.local_face_integrals),
      local_cell_integrals (sample_local_integrals.local_cell_integrals)
    {}



    /**
     * Actually do the computation based on the evaluated values and gradients in
     * ParallelData.
     */
    template <int dim>
    std::vector<float>
    DGErrorEstimator<dim>::integrate_over_face(DGParallelData<DoFHandler<dim> >              &parallel_data,
                                               const typename DoFHandler<dim>::face_iterator &face,
                                               const typename DoFHandler<dim>::active_cell_iterator       &cell,
                                               const typename DoFHandler<dim>::active_cell_iterator       &neighbor,
                                               const typename DoFHandler<dim>::active_cell_iterator       &helmholtz_cell,
                                               const typename DoFHandler<dim>::active_cell_iterator       &helmholtz_neighbor,
                                               FEFaceValues<dim>                             &fe_face_values_cell) const
    {
      const unsigned int    n_q_points          = parallel_data.temperature_face_values_cell.size();

      // multiply the temperature gradient on the face with the normal vector. Since we will
      // take the difference of gradients for internal faces, we may chose
      // the normal vector of one cell, taking that of the neighbor would only
      // change the sign. We take the outward normal. Scale by the diffusion values
      // on this side of the face.
      parallel_data.fe_values_cell.reinit(cell);
      parallel_data.fe_values_neighbor.reinit(neighbor);

      parallel_data.helmholtz_fe_values_cell.reinit(helmholtz_cell);
      parallel_data.helmholtz_fe_values_neighbor.reinit(helmholtz_neighbor);

      parallel_data.normal_vectors =
        fe_face_values_cell.get_present_fe_values().get_all_normal_vectors();

      for (unsigned int point=0; point<n_q_points; ++point)
        parallel_data.temperature_face_scaled_gradients_jump[point]
          = (parallel_data.temperature_face_gradients_cell[point] *
             parallel_data.normal_vectors[point] *
             parallel_data.diffusion_values_face_cell[point]);

      // calculate the jump of temperature values over the face

      for (unsigned int point=0; point<n_q_points; ++point)
        parallel_data.temperature_face_values_jump[point]
          = parallel_data.temperature_face_values_cell[point];


      if (face->at_boundary() == false)
        {
          //we use the convention that if neighbor == cell, then we don't really have a neighbor at all, and are on the boundary.
          Assert (cell != neighbor,
                  ExcInternalError());
          Assert (helmholtz_cell != helmholtz_neighbor,
                  ExcInternalError());

          // compute the jump in the scaled gradients by subtracting the neighboring face's scaled gradient

          for (unsigned int point=0; point<n_q_points; ++point)
            parallel_data.temperature_face_scaled_gradients_jump[point]
            -= (parallel_data.temperature_face_gradients_neighbor[point] *
                parallel_data.normal_vectors[point] *
                parallel_data.diffusion_values_face_neighbor[point]);

          // compute the jump in values

          for (unsigned int point=0; point<n_q_points; ++point)
            parallel_data.temperature_face_values_jump[point]
            -= parallel_data.temperature_face_values_neighbor[point];
        }


      if (face->at_boundary() == true)
        // if neumann boundary face, compute difference between normal derivative
        // and boundary function (which is zero Neumann)
        {
          //we use the convention that if neighbor == cell, then we don't really have a neighbor at all, and are on the boundary.
          Assert (cell == neighbor,
                  ExcInternalError());
          Assert (helmholtz_cell == helmholtz_neighbor,
                  ExcInternalError());

#if DEAL_II_VERSION_GTE(8,3,0)
          const types::boundary_id boundary_id = face->boundary_id();
#else
          const types::boundary_id boundary_id = face->boundary_indicator()
#endif

          if (this->get_parameters().fixed_temperature_boundary_indicators.find(boundary_id)
              == this->get_parameters().fixed_temperature_boundary_indicators.end())
            {
              //Neumann
              // Only zero Neumann conditions are allowed in ASPECT, so compute difference between
              // normal gradient and zero, i.e. leave as it is.
            }
          else
            {
              // face is Dirichlet face. Compute jump to Dirichlet function values

              for (unsigned int point=0; point<n_q_points; ++point)
                parallel_data.temperature_face_values_jump[point]
                -= this->get_boundary_temperature().boundary_temperature(//this->get_geometry_model(),
                     boundary_id,
                     fe_face_values_cell.quadrature_point(point));
            }
        }

      parallel_data.JxW_face_values
        = fe_face_values_cell.get_present_fe_values().get_JxW_values();

      double face_gradient_estimator_weight = 0;

      const unsigned int n_cell_q_points = parallel_data.value_H.size();

      std::vector<double> weight_values_edge (n_q_points);
      double weight_max_edge = 0;
      parallel_data.weighting_function->value_list (fe_face_values_cell.
                                                    get_present_fe_values().get_quadrature_points(),
                                                    parallel_data.face_value_H,
                                                    weight_values_edge);
      for (unsigned int q=0; q<n_q_points; ++q)
        weight_max_edge = std::max(weight_max_edge,
                                   weight_values_edge[q]);

      double weight_max_patch  = 0;
      double edge_varweight = 0;

      std::vector<double> addreac_cell     (n_cell_q_points, 1.);
      std::vector<double> addreac_neighbor (n_cell_q_points, 1.);
      std::vector<double> weight_values_cell      (n_cell_q_points);
      std::vector<double> weight_values_neighbor  (n_cell_q_points);
      std::vector<Tensor<1,dim,double> > grad_weight_values_cell (n_cell_q_points,
                                                                  Tensor<1,dim,double>());
      std::vector<Tensor<1,dim,double> > grad_weight_values_neighbor (n_cell_q_points,
                                                                      Tensor<1,dim,double>());

      std::vector<double> L_cell (n_cell_q_points);
      std::vector<double> L_neighbor (n_cell_q_points);
      double lambda_sq_cell              =  std::numeric_limits<double>::signaling_NaN();
      double lambda_sq_neighbor          =  std::numeric_limits<double>::signaling_NaN();

      double grad_helmholtz_sq_max       = -std::numeric_limits<double>::max();
      double weight_cell_max             = -std::numeric_limits<double>::max();
      double weight_neighbor_max         = -std::numeric_limits<double>::max();
      double weight_cell_min             =  std::numeric_limits<double>::max();
      double weight_neighbor_min         =  std::numeric_limits<double>::max();
      double grad_weight_sq_cell_max     = -std::numeric_limits<double>::max();
      double grad_weight_sq_neighbor_max = -std::numeric_limits<double>::max();
      double L_cell_min                  =  std::numeric_limits<double>::max();
      double L_neighbor_min              =  std::numeric_limits<double>::max();
      double weighted_L_patch_max        = -std::numeric_limits<double>::max();
      double cell_weight_squared         =  std::numeric_limits<double>::signaling_NaN();
      double neighbor_weight_squared     =  std::numeric_limits<double>::signaling_NaN();

      {
        //calculate max_edge (grad H)^2
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            grad_helmholtz_sq_max = std::max(grad_helmholtz_sq_max,
                                             parallel_data.face_gradient_H[q] * parallel_data.face_gradient_H[q]);
            Assert (numbers::is_finite(grad_helmholtz_sq_max),
                    ExcMessage ("grad_helmholtz_sq_max is not finite"));
            Assert (grad_helmholtz_sq_max >= 0,
                    ExcMessage ("grad_helmholtz_sq_max is negative"));
          }

        parallel_data.helmholtz_fe_values_cell.get_present_fe_values()
        .get_function_values (parallel_data.helmholtz_solution, parallel_data.value_H);
        parallel_data.helmholtz_fe_values_cell.get_present_fe_values()
        .get_function_gradients (parallel_data.helmholtz_solution, parallel_data.gradient_H);
        parallel_data.fe_values_cell.get_present_fe_values()[this->introspection().extractors.velocities]
        .get_function_values (this->get_solution(), parallel_data.velocity_values);
        parallel_data.fe_values_cell.get_present_fe_values()[this->introspection().extractors.velocities]
        .get_function_divergences (this->get_solution(), parallel_data.div_u_cell);

        {
          std::vector<std::vector<double> > prelim_composition_values (this->n_compositional_fields(),
                                                                       std::vector<double> (parallel_data.cell_quadrature.size()));

          MaterialModel::MaterialModelInputs<dim> in(parallel_data.cell_quadrature.size(),
                                                     this->n_compositional_fields());
          MaterialModel::MaterialModelOutputs<dim> out(parallel_data.cell_quadrature.size(),
                                                       this->n_compositional_fields());

          parallel_data.fe_values_cell.get_present_fe_values()[this->introspection().extractors.temperature]
          .get_function_values (this->get_solution(), in.temperature);

          in.position = parallel_data.fe_values_cell.get_quadrature_points();
          for (unsigned int i=0; i<parallel_data.cell_quadrature.size(); ++i)
            {
              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                in.composition[i][c] = prelim_composition_values[c][i];
            }

          in.cell = &cell;

          this->get_material_model().evaluate(in, out);

          HeatingModel::HeatingModelOutputs heating_model_outputs(parallel_data.cell_quadrature.size(), this->get_parameters().n_compositional_fields);
          this->get_heating_model_manager().evaluate(in,
                                                     out,
                                                     heating_model_outputs);


          for (unsigned int q=0; q<parallel_data.cell_quadrature.size(); ++q)
            {
              parallel_data.diffusion_values_cell[q] = out.thermal_conductivities[q] / (out.densities[q] * out.specific_heat[q] + heating_model_outputs.lhs_latent_heat_terms[q]);
              Assert (numbers::is_finite(parallel_data.diffusion_values_cell[q]),
                      ExcMessage ("parallel_data.diffusion_values_cell[q] is not finite"));
            }
        }

        parallel_data.weighting_function->value_list (parallel_data.fe_values_cell.
                                                      get_present_fe_values().get_quadrature_points(),
                                                      parallel_data.value_H,
                                                      weight_values_cell);
        parallel_data.weighting_function->gradient_list (parallel_data.fe_values_cell.
                                                         get_present_fe_values().get_quadrature_points(),
                                                         parallel_data.value_H,
                                                         parallel_data.gradient_H,
                                                         grad_weight_values_cell);
        parallel_data.reaction_function->value_list(parallel_data.fe_values_cell.
                                                    get_present_fe_values().get_quadrature_points(),
                                                    parallel_data.div_u_cell,
                                                    parallel_data.diffusion_values_cell,
                                                    parallel_data.gradient_H,
                                                    parallel_data.velocity_values,
                                                    addreac_cell);


        //loop over points, calculating L at each point.
        for (unsigned int q=0; q<n_cell_q_points; ++q)
          {
            L_cell[q] = addreac_cell[q] + 1./2. * (EquationData::alpha * parallel_data.gradient_H[q]*parallel_data.velocity_values[q]
                                                   -
                                                   (1-EquationData::alpha * parallel_data.diffusion_values_cell[q]) * parallel_data.div_u_cell[q]
                                                   -
                                                   EquationData::alpha * EquationData::alpha * parallel_data.diffusion_values_cell[q] *parallel_data.gradient_H[q] * parallel_data.gradient_H[q]);
            Assert (numbers::is_finite(L_cell[q]),
                    ExcMessage("L_cell[q] is not finite"));
            Assert (L_cell[q] >= 0,
                    ExcMessage("L_cell[q] is negative"));
          }
        //find maxes and mins of each value as appropriate
        for (unsigned int q=0; q<n_cell_q_points; ++q)
          {
            weight_cell_max = std::max(weight_cell_max,
                                       weight_values_cell[q]);
            weight_cell_min = std::min(weight_cell_min,
                                       weight_values_cell[q]);
            grad_weight_sq_cell_max = std::max(grad_weight_sq_cell_max,
                                               grad_weight_values_cell[q].norm());
            L_cell_min = std::min(L_cell_min,
                                  L_cell[q]);
            weighted_L_patch_max = std::max(weighted_L_patch_max,
                                            L_cell[q]*weight_values_cell[q]);
            Assert (numbers::is_finite(weight_cell_max),
                    ExcMessage("weight_cell_max is not finite"));
            Assert (weight_cell_max >= 0,
                    ExcMessage("weight_cell_max is negative"));
            Assert (numbers::is_finite(weight_cell_min),
                    ExcMessage("weight_cell_min is not finite"));
            Assert (weight_cell_min > 0,
                    ExcMessage("weight_cell_min is non-positive"));
            Assert (numbers::is_finite(grad_weight_sq_cell_max),
                    ExcMessage("grad_weight_sq_cell_max is not finite"));
            Assert (grad_weight_sq_cell_max >= 0,
                    ExcMessage("grad_weight_sq_cell_max is negative"));
            Assert (numbers::is_finite(L_cell_min),
                    ExcMessage("L_cell_min is not finite"));
            Assert (L_cell_min > 0,
                    ExcMessage("L_cell_min is not positive"));
            Assert (numbers::is_finite(weighted_L_patch_max),
                    ExcMessage("weighted_L_patch_max is not finite"));
            Assert (weighted_L_patch_max >= 0,
                    ExcMessage("weighted_L_patch_max is negative"));
          }

        for (unsigned int q=0; q<n_cell_q_points; ++q)
          {
            lambda_sq_cell = std::max(grad_weight_sq_cell_max/L_cell_min,
                                      weight_cell_max*weight_cell_max/parallel_data.diffusion_values_cell[q]);
            Assert (numbers::is_finite(lambda_sq_cell),
                    ExcMessage("lambda_sq_cell is not finite"));
            Assert (lambda_sq_cell >= 0,
                    ExcMessage("lambda_sq_cell is negative"));
          }

        if (face->at_boundary() == false)
          {
            //Do calculations on neighboring cell
            parallel_data.fe_values_neighbor.reinit (neighbor);

            parallel_data.helmholtz_fe_values_neighbor.reinit (helmholtz_neighbor);

            parallel_data.helmholtz_fe_values_neighbor.get_present_fe_values()
            .get_function_values (parallel_data.helmholtz_solution, parallel_data.value_H_neighbor);
            parallel_data.helmholtz_fe_values_neighbor.get_present_fe_values()
            .get_function_gradients (parallel_data.helmholtz_solution, parallel_data.gradient_H_neighbor);
            parallel_data.fe_values_neighbor.get_present_fe_values()[this->introspection().extractors.velocities]
            .get_function_values (this->get_solution(), parallel_data.velocity_values_neighbor);
            parallel_data.fe_values_neighbor.get_present_fe_values()[this->introspection().extractors.velocities]
            .get_function_divergences (this->get_solution(), parallel_data.div_u_neighbor);

            std::vector<std::vector<double> > prelim_composition_values (this->n_compositional_fields(),
                                                                         std::vector<double> (parallel_data.cell_quadrature.size()));

            MaterialModel::MaterialModelInputs<dim> in(parallel_data.cell_quadrature.size(),
                                                       this->n_compositional_fields());
            MaterialModel::MaterialModelOutputs<dim> out(parallel_data.cell_quadrature.size(),
                                                         this->n_compositional_fields());

            parallel_data.fe_values_neighbor.get_present_fe_values()[this->introspection().extractors.temperature]
            .get_function_values (this->get_solution(), in.temperature);

            in.position = parallel_data.fe_values_neighbor.get_quadrature_points();
            for (unsigned int i=0; i<parallel_data.cell_quadrature.size(); ++i)
              {
                for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                  in.composition[i][c] = prelim_composition_values[c][i];
              }

            in.cell = &neighbor;

            this->get_material_model().evaluate(in, out);

            HeatingModel::HeatingModelOutputs heating_model_outputs(parallel_data.cell_quadrature.size(), this->get_parameters().n_compositional_fields);
            this->get_heating_model_manager().evaluate(in,
                                                       out,
                                                       heating_model_outputs);


            for (unsigned int q=0; q<parallel_data.cell_quadrature.size(); ++q)
              {
                parallel_data.diffusion_values_neighbor[q] = out.thermal_conductivities[q] / (out.densities[q] * out.specific_heat[q] + heating_model_outputs.lhs_latent_heat_terms[q]);
                Assert (numbers::is_finite(parallel_data.diffusion_values_neighbor[q]),
                        ExcMessage ("parallel_data.diffusion_values_neighbor[q] is not finite"));
              }


            parallel_data.weighting_function->value_list (parallel_data.fe_values_neighbor.
                                                          get_present_fe_values().get_quadrature_points(),
                                                          parallel_data.value_H_neighbor,
                                                          weight_values_neighbor);
            parallel_data.weighting_function->gradient_list (parallel_data.fe_values_neighbor.
                                                             get_present_fe_values().get_quadrature_points(),
                                                             parallel_data.value_H_neighbor,
                                                             parallel_data.gradient_H_neighbor,
                                                             grad_weight_values_neighbor);
            parallel_data.reaction_function->value_list(parallel_data.fe_values_neighbor.
                                                        get_present_fe_values().get_quadrature_points(),
                                                        parallel_data.div_u_neighbor,
                                                        parallel_data.diffusion_values_neighbor,
                                                        parallel_data.gradient_H_neighbor,
                                                        parallel_data.velocity_values_neighbor,
                                                        addreac_neighbor);

            //loop over points, calculating L at each point.
            for (unsigned int q=0; q<n_cell_q_points; ++q)
              {
                L_neighbor[q] = addreac_neighbor[q] + 1./2. * (EquationData::alpha * parallel_data.gradient_H_neighbor[q]*parallel_data.velocity_values_neighbor[q]
                                                               -
                                                               (1-EquationData::alpha * parallel_data.diffusion_values_neighbor[0]) * parallel_data.div_u_neighbor[q]
                                                               -
                                                               EquationData::alpha * EquationData::alpha * parallel_data.diffusion_values_neighbor[q] *parallel_data.gradient_H_neighbor[q] * parallel_data.gradient_H_neighbor[q]);
                Assert (numbers::is_finite(L_neighbor[q]),
                        ExcMessage("L_neighbor[q] is not finite"));
                Assert (L_neighbor[q] >= 0,
                        ExcMessage("L_neighbor[q] is negative"));
              }
            //find maxes and mins of each value as appropriate
            for (unsigned int q=0; q<n_cell_q_points; ++q)
              {
                weight_neighbor_max = std::max(weight_neighbor_max,
                                               weight_values_neighbor[q]);
                weight_neighbor_min = std::min(weight_neighbor_min,
                                               weight_values_neighbor[q]);
                grad_weight_sq_neighbor_max = std::max(grad_weight_sq_neighbor_max,
                                                       grad_weight_values_neighbor[q].norm());
                L_neighbor_min = std::min(L_neighbor_min,
                                          L_neighbor[q]);
                weighted_L_patch_max = std::max(weighted_L_patch_max,
                                                L_neighbor[q]*weight_values_neighbor[q]);
              }
            Assert (numbers::is_finite(weight_neighbor_max),
                    ExcMessage("weight_neighbor_max is not finite"))
            Assert (weight_neighbor_max >= 0,
                    ExcMessage("weight_neighbor_max is negative"))
            Assert (numbers::is_finite(weight_neighbor_min),
                    ExcMessage("weight_neighbor_min is not finite"))
            Assert (weight_neighbor_min >= 0,
                    ExcMessage("weight_neighbor_min is negative"))
            Assert (numbers::is_finite(grad_weight_sq_neighbor_max),
                    ExcMessage("grad_weight_sq_neighbor_max is not finite"))
            Assert (grad_weight_sq_neighbor_max >= 0,
                    ExcMessage("grad_weight_sq_neighbor_max is negative"))
            Assert (numbers::is_finite(L_neighbor_min),
                    ExcMessage("L_neighbor_min is not finite"))
            Assert (L_neighbor_min >= 0,
                    ExcMessage("L_neighbor_min is negative"))
            Assert (numbers::is_finite(weighted_L_patch_max),
                    ExcMessage("weighted_L_patch_max is not finite"))
            Assert (weighted_L_patch_max >= 0,
                    ExcMessage("weighted_L_patch_max is negative"))

            for (unsigned int q=0; q<n_cell_q_points; ++q)
              {
                lambda_sq_neighbor = std::max(grad_weight_sq_neighbor_max/L_neighbor_min,
                                              weight_neighbor_max*weight_neighbor_max/parallel_data.diffusion_values_neighbor[q]);
              }
            Assert (numbers::is_finite(lambda_sq_neighbor),
                    ExcMessage("lambda_sq_neighbor is not finite"))
            Assert (lambda_sq_neighbor >= 0,
                    ExcMessage("lambda_sq_neighbor is negative"))
          }

        if (face->at_boundary() == false)
          {
            cell_weight_squared = weight_cell_min*std::min(weight_cell_max*weight_cell_max/L_cell_min,
                                                           cell->diameter()*cell->diameter()*lambda_sq_cell);
            neighbor_weight_squared = weight_neighbor_min*std::min(weight_neighbor_max*weight_neighbor_max/L_neighbor_min,
                                                                   neighbor->diameter()*neighbor->diameter()*lambda_sq_neighbor);
            face_gradient_estimator_weight = std::min(cell->diameter() * lambda_sq_cell/ weight_cell_min,
                                                      neighbor->diameter() * lambda_sq_neighbor/ weight_neighbor_min);
            weight_max_patch = std::max(weight_cell_max,
                                        weight_neighbor_max);
            edge_varweight = std::sqrt(std::max(cell_weight_squared,
                                                neighbor_weight_squared));
          }
        else
          {
            cell_weight_squared = weight_cell_min*std::min(weight_cell_max*weight_cell_max/L_cell_min,
                                                           cell->diameter()*cell->diameter()*lambda_sq_cell);
            face_gradient_estimator_weight= cell->diameter() * lambda_sq_cell/ weight_cell_min;
            weight_max_patch = weight_cell_max;
            edge_varweight= std::sqrt(cell_weight_squared);
          }
      }

      double edge_patch_max_velocity_squared = 0;
      double edge_patch_max_modified_velocity_squared = 0;
      for (unsigned int p=0; p<n_cell_q_points; ++p)
        {
          edge_patch_max_velocity_squared = std::max(edge_patch_max_velocity_squared,
                                                     parallel_data.velocity_values[p].norm());
          edge_patch_max_modified_velocity_squared = std::max(edge_patch_max_modified_velocity_squared,
                                                              (parallel_data.velocity_values[p]-EquationData::alpha*parallel_data.diffusion_values_cell[p]*parallel_data.gradient_H[p]).norm());
        }

      double face_value_estimator_weight = this->get_parameters().discontinuous_penalty
                                           * this->get_parameters().temperature_degree
                                           * this->get_parameters().temperature_degree/face->diameter() * parallel_data.diffusion_values_cell[0]
                                           * (weight_max_patch
                                              + edge_varweight*this->get_parameters().discontinuous_penalty
                                              * this->get_parameters().temperature_degree
                                              * this->get_parameters().temperature_degree*parallel_data.diffusion_values_cell[0]
                                              + weight_max_edge*EquationData::alpha*EquationData::alpha*parallel_data.diffusion_values_cell[0]*grad_helmholtz_sq_max/std::min(L_cell_min,L_neighbor_min))
                                           + edge_varweight * edge_patch_max_velocity_squared
                                           + face->diameter() * std::sqrt(weighted_L_patch_max)
                                           + (weight_max_patch * face->diameter() / parallel_data.diffusion_values_cell[0]) * edge_patch_max_modified_velocity_squared;

      // take the square of the scaled gradient jump[i] for integration, and sum up. Do the same for value jump[i]
      std::vector<float> face_integral (parallel_data.n_face_terms, 0);
      for (unsigned int p=0; p<n_q_points; ++p)
        {
          Assert (numbers::is_finite(face_gradient_estimator_weight
                                     * numbers::NumberTraits<float>::abs_square(parallel_data.temperature_face_scaled_gradients_jump[p])
                                     * parallel_data.JxW_face_values[p]),
                  ExcMessage("Face gradient contribution is not finite"));
          Assert (face_gradient_estimator_weight
                  * numbers::NumberTraits<float>::abs_square(parallel_data.temperature_face_scaled_gradients_jump[p])
                  * parallel_data.JxW_face_values[p] >= 0,
                  ExcMessage("Face gradient contribution is nagetive"));
          Assert (numbers::is_finite(face_value_estimator_weight
                                     * numbers::NumberTraits<float>::abs_square(parallel_data.temperature_face_values_jump[p])
                                     * parallel_data.JxW_face_values[p]),
                  ExcMessage("Face value contribution is not finite"));
          Assert (face_value_estimator_weight
                  * numbers::NumberTraits<float>::abs_square(parallel_data.temperature_face_values_jump[p])
                  * parallel_data.JxW_face_values[p] >= 0,
                  ExcMessage("Face value contribution is negative"));

          face_integral[0] += face_gradient_estimator_weight
                              * numbers::NumberTraits<float>::abs_square(parallel_data.temperature_face_scaled_gradients_jump[p])
                              * parallel_data.JxW_face_values[p];
          face_integral[1] += face_value_estimator_weight
                              * numbers::NumberTraits<float>::abs_square(parallel_data.temperature_face_values_jump[p])
                              * parallel_data.JxW_face_values[p];
        }

      return face_integral;
    }


    template <int dim>
    void
    DGErrorEstimator<dim>::integrate_over_regular_face (std::map<typename DoFHandler<dim>::face_iterator,std::vector<float> > &local_face_integrals,
                                                        const typename DoFHandler<dim>::active_cell_iterator                  &cell,
                                                        const unsigned int                                                     face_no,
                                                        const typename DoFHandler<dim>::active_cell_iterator                  &helmholtz_cell,
                                                        DGParallelData<DoFHandler<dim> >                                      &parallel_data,
                                                        FEFaceValues<dim>                                                     &fe_face_values_cell,
                                                        FEFaceValues<dim>                                                     &fe_face_values_neighbor) const
    {
      const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
      parallel_data.fe_face_values_cell.reinit(cell, face_no);

      // the values of the compositional fields are stored as blockvectors for each field
      // we have to extract them in this structure
      std::vector<std::vector<double> > prelim_composition_values_cell (this->n_compositional_fields(),
                                                                        std::vector<double> (parallel_data.face_quadrature.size()));

      MaterialModel::MaterialModelInputs<dim> in_cell(parallel_data.face_quadrature.size(),
                                                      this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out_cell(parallel_data.face_quadrature.size(),
                                                        this->n_compositional_fields());

      // get values of the finite element
      // function on this cell's face
      fe_face_values_cell[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
          parallel_data.temperature_face_values_cell);
      in_cell.temperature = parallel_data.temperature_face_values_cell;

      // store the gradient of the solution on this cell's face
      fe_face_values_cell[this->introspection().extractors.temperature].get_function_gradients (this->get_solution(),
          parallel_data.temperature_face_gradients_cell);

      //velocity is only required on cell face as it is a continuous field - using neighbor would give us the same results
      fe_face_values_cell[this->introspection().extractors.velocities].get_function_values(this->get_solution(),
          parallel_data.velocity_face_values_cell);

      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        fe_face_values_cell[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
            prelim_composition_values_cell[c]);

      in_cell.position = fe_face_values_cell.get_quadrature_points();
      for (unsigned int i=0; i<parallel_data.face_quadrature.size(); ++i)
        {
          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            in_cell.composition[i][c] = prelim_composition_values_cell[c][i];
        }

      in_cell.cell = &cell;

      this->get_material_model().evaluate(in_cell, out_cell);

      HeatingModel::HeatingModelOutputs heating_model_outputs_cell(parallel_data.face_quadrature.size(), this->get_parameters().n_compositional_fields);
      this->get_heating_model_manager().evaluate(in_cell,
                                                 out_cell,
                                                 heating_model_outputs_cell);

      // All face data has now been evaluated. Now compute the diffusion at each point.

      AssertThrow (parallel_data.face_quadrature.size() == parallel_data.diffusion_values_face_cell.size(),
                   ExcMessage("diffusion_values_face_cell is not same size as face_quadrature"));

      for (unsigned int q=0; q< parallel_data.face_quadrature.size(); ++q)
        parallel_data.diffusion_values_face_cell[q] = out_cell.thermal_conductivities[q] /
                                                      (out_cell.densities[q] * out_cell.specific_heat[q]
                                                       + heating_model_outputs_cell.lhs_latent_heat_terms[q]);

      //This is JUST so we can get the velocities on the cell and neighbor.
      parallel_data.fe_values_cell.reinit(cell);
      parallel_data.fe_values_cell[this->introspection().extractors.velocities].get_function_values(this->get_solution(),
          parallel_data.velocity_values);

      typename DoFHandler<dim>::active_cell_iterator neighbor;
      typename DoFHandler<dim>::active_cell_iterator helmholtz_neighbor;
      // Now move on to the neighbor
      if (face->at_boundary() == false)
        {
          Assert (cell->neighbor(face_no).state() == IteratorState::valid,
                  ExcInternalError());
          Assert (helmholtz_cell->neighbor(face_no).state() == IteratorState::valid,
                  ExcInternalError());

          neighbor = cell->neighbor(face_no);
          helmholtz_neighbor = helmholtz_cell->neighbor(face_no);

          // find which number the current face has relative to the
          // neighboring cell
          const unsigned int neighbor_neighbor
            = cell->neighbor_of_neighbor (face_no);
          Assert (neighbor_neighbor<GeometryInfo<dim>::faces_per_cell,
                  ExcInternalError());

          // get restriction of finite element function of @p{neighbor} to the
          // common face.
          fe_face_values_neighbor.reinit (neighbor, neighbor_neighbor);

          std::vector<std::vector<double> > prelim_composition_values_neighbor (this->n_compositional_fields(),
                                                                                std::vector<double> (parallel_data.face_quadrature.size()));

          MaterialModel::MaterialModelInputs<dim> in_neighbor(parallel_data.face_quadrature.size(),
                                                              this->n_compositional_fields());
          MaterialModel::MaterialModelOutputs<dim> out_neighbor(parallel_data.face_quadrature.size(),
                                                                this->n_compositional_fields());


          fe_face_values_neighbor[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
              parallel_data.temperature_face_values_neighbor);
          in_neighbor.temperature = parallel_data.temperature_face_values_neighbor;

          fe_face_values_neighbor[this->introspection().extractors.temperature].get_function_gradients (this->get_solution(),
              parallel_data.temperature_face_gradients_neighbor);

          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            fe_face_values_neighbor[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                prelim_composition_values_neighbor[c]);

          in_neighbor.position = fe_face_values_neighbor.get_quadrature_points();
          for (unsigned int i=0; i<parallel_data.face_quadrature.size(); ++i)
            {
              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                in_neighbor.composition[i][c] = prelim_composition_values_neighbor[c][i];
            }

          in_neighbor.cell = &neighbor;

          this->get_material_model().evaluate(in_neighbor, out_neighbor);

          HeatingModel::HeatingModelOutputs heating_model_outputs_neighbor(parallel_data.face_quadrature.size(), this->get_parameters().n_compositional_fields);
          this->get_heating_model_manager().evaluate(in_neighbor,
                                                     out_neighbor,
                                                     heating_model_outputs_neighbor);

          // All neighbor data has now been evaluated. Now compute the diffusion at each point.

          AssertThrow (parallel_data.face_quadrature.size() == parallel_data.diffusion_values_face_neighbor.size(),
                       ExcMessage("diffusion_values_face_neighbor is not same size as face_quadrature"));

          for (unsigned int q=0; q< parallel_data.face_quadrature.size(); ++q)
            parallel_data.diffusion_values_face_neighbor[q] = out_neighbor.thermal_conductivities[q] /
                                                              (out_neighbor.densities[q] * out_neighbor.specific_heat[q]
                                                               + heating_model_outputs_neighbor.lhs_latent_heat_terms[q]);
        }
      else
        /* at_boundary() == true, so it is at boundary. No action required, as the values of
         * Dirichlet BC will be read inside integrate_over_face, and Neumann BCs will require no data.
         */
        {
          neighbor = cell; //we use the convention that if neighbor == cell, then we don't really have a neighbor at all, and are on the boundary.
          helmholtz_neighbor = helmholtz_cell;
          local_face_integrals[face] = std::vector<float>(parallel_data.n_face_terms);
        }

      // now go to the generic function that does all the other things
      local_face_integrals[face] =
      integrate_over_face (parallel_data, face, cell, neighbor, helmholtz_cell, helmholtz_neighbor, fe_face_values_cell);

      /*local_face_integrals[face] now contains
       * int( [grad conv]^2 ) in element 0 and int( [conv]^2 ) in element 1,
       * where the integration is over the face
       */
    }


    template <int dim>
    void
    DGErrorEstimator<dim>::integrate_over_irregular_face (std::map<typename DoFHandler<dim>::face_iterator,std::vector<float> > &local_face_integrals,
                                                          const typename DoFHandler<dim>::active_cell_iterator                  &cell,
                                                          const unsigned int                                                     face_no,
                                                          const typename DoFHandler<dim>::active_cell_iterator                  &helmholtz_cell,
                                                          DGParallelData<DoFHandler<dim> >                                      &parallel_data,
                                                          FEFaceValues<dim>                                                     &fe_face_values,
                                                          FESubfaceValues<dim>                                                  &fe_subface_values) const
    {
      const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);

      const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
      const typename DoFHandler<dim>::cell_iterator helmholtz_neighbor = helmholtz_cell->neighbor(face_no);

      Assert (neighbor.state() == IteratorState::valid, ExcInternalError());
      Assert (helmholtz_neighbor.state() == IteratorState::valid, ExcInternalError());
      Assert (face->has_children(), ExcInternalError());

      // set up a vector of the gradients of the finite element function on
      // this cell at the quadrature points
      //
      // let psi be a short name for [a grad u_h], where the second index be
      // the component of the finite element, and the first index the number
      // of the quadrature point

      // store which number cell has in the list of neighbors of
      // @p{neighbor}
      const unsigned int neighbor_neighbor
        = cell->neighbor_of_neighbor (face_no);
      Assert (neighbor_neighbor<GeometryInfo<dim>::faces_per_cell,
              ExcInternalError());

      // loop over all subfaces
      for (unsigned int subface_no=0; subface_no<face->n_children(); ++subface_no)
        {
          // get an iterator pointing to the cell behind the present subface
          const typename DoFHandler<dim>::active_cell_iterator neighbor_child
            = cell->neighbor_child_on_subface (face_no, subface_no);
          Assert (!neighbor_child->has_children(),
                  ExcInternalError());
          const typename DoFHandler<dim>::active_cell_iterator helmholtz_neighbor_child
            = helmholtz_cell->neighbor_child_on_subface (face_no, subface_no);
          Assert (!helmholtz_neighbor_child->has_children(),
                  ExcInternalError());

          // restrict the finite element on the present cell to the subface
          fe_subface_values.reinit (cell, face_no, subface_no);

          // the values of the compositional fields are stored as blockvectors for each field
          // we have to extract them in this structure
          std::vector<std::vector<double> > prelim_composition_values_cell (this->n_compositional_fields(),
                                                                            std::vector<double> (parallel_data.face_quadrature.size()));

          MaterialModel::MaterialModelInputs<dim> in_cell(parallel_data.face_quadrature.size(),
                                                          this->n_compositional_fields());
          MaterialModel::MaterialModelOutputs<dim> out_cell(parallel_data.face_quadrature.size(),
                                                            this->n_compositional_fields());

          // get values of the finite element
          // function on this cell's face
          fe_subface_values[this->introspection().extractors.temperature]
          .get_function_values (this->get_solution(), parallel_data.temperature_face_values_cell);
          in_cell.temperature = parallel_data.temperature_face_values_cell;

          // store the gradient of the solution on this cell's face
          fe_subface_values[this->introspection().extractors.temperature]
          .get_function_gradients (this->get_solution(), parallel_data.temperature_face_gradients_cell);

          //velocity is only required on cell face as it is a continuous field - using neighbor would give us the same results
          fe_subface_values[this->introspection().extractors.velocities]
          .get_function_values (this->get_solution(), parallel_data.velocity_face_values_cell);

          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            fe_subface_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                prelim_composition_values_cell[c]);

          in_cell.position = fe_subface_values.get_quadrature_points();
          for (unsigned int i=0; i<parallel_data.face_quadrature.size(); ++i)
            {
              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                in_cell.composition[i][c] = prelim_composition_values_cell[c][i];
            }

          in_cell.cell = &cell;

          this->get_material_model().evaluate(in_cell, out_cell);

          HeatingModel::HeatingModelOutputs heating_model_outputs_cell(parallel_data.face_quadrature.size(), this->get_parameters().n_compositional_fields);
          this->get_heating_model_manager().evaluate(in_cell,
                                                     out_cell,
                                                     heating_model_outputs_cell);

          // All cell data has now been evaluated. Now compute the diffusion at each point.

          AssertThrow (parallel_data.face_quadrature.size() == parallel_data.diffusion_values_face_cell.size(),
                       ExcMessage("diffusion_values_face_cell is not same size as face_quadrature"));

          for (unsigned int q=0; q< parallel_data.face_quadrature.size(); ++q)
            parallel_data.diffusion_values_face_cell[q] = out_cell.thermal_conductivities[q] /
                                                          (out_cell.densities[q] * out_cell.specific_heat[q]
                                                           + heating_model_outputs_cell.lhs_latent_heat_terms[q]);

          // Now move on to the neighbor

          // restrict the finite element on the neighbor cell to the common
          // subface.
          fe_face_values.reinit (neighbor_child, neighbor_neighbor);

          std::vector<std::vector<double> > prelim_composition_values_neighbor (this->n_compositional_fields(),
                                                                                std::vector<double> (parallel_data.face_quadrature.size()));

          MaterialModel::MaterialModelInputs<dim> in_neighbor(parallel_data.face_quadrature.size(),
                                                              this->n_compositional_fields());
          MaterialModel::MaterialModelOutputs<dim> out_neighbor(parallel_data.face_quadrature.size(),
                                                                this->n_compositional_fields());

          // get values of the finite element
          // function on the neighbor's face
          fe_face_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
              parallel_data.temperature_face_values_neighbor);
          in_neighbor.temperature = parallel_data.temperature_face_values_neighbor;

          // store the gradient of the solution on the neighbor's face
          fe_face_values[this->introspection().extractors.temperature].get_function_gradients (this->get_solution(),
              parallel_data.temperature_face_gradients_neighbor);

          //since velocity field is continuous, no velocity_face-values_neighbor needed - tit is identical

          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            fe_face_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                prelim_composition_values_neighbor[c]);

          in_neighbor.position = fe_face_values.get_quadrature_points();
          for (unsigned int i=0; i<parallel_data.face_quadrature.size(); ++i)
            {
              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                in_neighbor.composition[i][c] = prelim_composition_values_neighbor[c][i];
            }

          in_neighbor.cell = &neighbor_child;

          this->get_material_model().evaluate(in_neighbor, out_neighbor);

          HeatingModel::HeatingModelOutputs heating_model_outputs_neighbor(parallel_data.face_quadrature.size(), this->get_parameters().n_compositional_fields);
          this->get_heating_model_manager().evaluate(in_neighbor,
                                                     out_neighbor,
                                                     heating_model_outputs_neighbor);

          // All neighbor's data has now been evaluated. Now compute the diffusion at each point.

          AssertThrow (parallel_data.face_quadrature.size() == parallel_data.diffusion_values_face_neighbor.size(),
                       ExcMessage("diffusion_values_face_neighbor is not same size as face_quadrature"));

          for (unsigned int q=0; q< parallel_data.face_quadrature.size(); ++q)
            parallel_data.diffusion_values_face_neighbor[q] = out_neighbor.thermal_conductivities[q] /
                                                              (out_neighbor.densities[q] * out_neighbor.specific_heat[q]
                                                               + heating_model_outputs_neighbor.lhs_latent_heat_terms[q]);

          // call generic evaluate function
          local_face_integrals[neighbor_child->face(neighbor_neighbor)] =
          integrate_over_face (parallel_data, face, cell, neighbor_child, helmholtz_cell, helmholtz_neighbor_child, fe_face_values);
        }

      // finally loop over all subfaces to collect the contributions of the
      // subfaces and store them with the mother face
      std::vector<float> sum (parallel_data.n_face_terms);
      for (unsigned int subface_no=0; subface_no<face->n_children(); ++subface_no)
        for (unsigned int face_term=0; face_term<parallel_data.n_face_terms; ++face_term)
          {
            Assert (local_face_integrals.find(face->child(subface_no)) !=
                    local_face_integrals.end(),
                    ExcInternalError());
            Assert (local_face_integrals[face->child(subface_no)][face_term] >= 0,
                    ExcInternalError());

            sum[face_term] += local_face_integrals[face->child(subface_no)][face_term];
          }

      local_face_integrals[face] = sum;

      /*local_face_integrals[face] now contains
       * int( [grad conv]^2 ) in element 0 and int( [conv]^2 ) in element 1,
       * where the integration is over the face
       */
    }


    template <int dim>
    void
    DGErrorEstimator<dim>::integrate_over_cell(std::map<typename DoFHandler<dim>::active_cell_iterator,std::vector<float> > &local_cell_integrals,
                                               typename DoFHandler<dim>::active_cell_iterator                               &cell,
                                               DGParallelData<DoFHandler<dim> >                                             &parallel_data,
                                               typename DoFHandler<dim>::active_cell_iterator                               &projection_cell,
                                               FEValues<dim>                                                                &projection_fe_values,
                                               Quadrature<dim>                                                              &projection_quadrature,
                                               FEValues<dim>                                                                &helmholtz_fe_values,
                                               typename DoFHandler<dim>::active_cell_iterator                               &helmholtz_cell) const
    {
      std::vector<float> cell_integral (parallel_data.n_cell_terms);

      parallel_data.fe_values_cell.reinit(cell);

      // the values of the compositional fields are stored as blockvectors for each field
      // we have to extract them in this structure
      std::vector<std::vector<double> > prelim_composition_values (this->n_compositional_fields(),
                                                                   std::vector<double> (parallel_data.cell_quadrature.size()));

      // The projection FE has only one component, so extract that
      FEValuesExtractors::Scalar extract_projection(0);

      const unsigned int  n_q_points    = projection_fe_values.n_quadrature_points;

      // extract necessary evaluations for assembly of indicator
      MaterialModel::MaterialModelInputs<dim> in(parallel_data.cell_quadrature.size(),
                                                 this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(parallel_data.cell_quadrature.size(),
                                                   this->n_compositional_fields());


      parallel_data.fe_values_cell[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
          parallel_data.temperature_values);
      in.temperature = parallel_data.temperature_values;
      parallel_data.fe_values_cell[this->introspection().extractors.temperature].get_function_values (this->get_old_solution(),
          parallel_data.old_temperature_values);
      parallel_data.fe_values_cell[this->introspection().extractors.temperature].get_function_gradients (this->get_solution(),
          parallel_data.temperature_gradients);
      parallel_data.fe_values_cell[this->introspection().extractors.temperature].get_function_laplacians (this->get_solution(),
          parallel_data.temperature_laplacians);
      parallel_data.fe_values_cell[this->introspection().extractors.velocities].get_function_values(this->get_solution(),
          parallel_data.velocity_values);
      parallel_data.fe_values_cell[this->introspection().extractors.velocities].get_function_divergences(this->get_solution(),
          parallel_data.div_u_cell);

      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        parallel_data.fe_values_cell[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
            prelim_composition_values[c]);

      in.position = parallel_data.fe_values_cell.get_quadrature_points();
      for (unsigned int i=0; i<parallel_data.cell_quadrature.size(); ++i)
        {
          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            in.composition[i][c] = prelim_composition_values[c][i];
        }

      in.cell = &cell;

      this->get_material_model().evaluate(in, out);

      HeatingModel::HeatingModelOutputs heating_model_outputs(parallel_data.cell_quadrature.size(), this->get_parameters().n_compositional_fields);
      this->get_heating_model_manager().evaluate(in,
                                                 out,
                                                 heating_model_outputs);

      double velocity_grad_temperature = 0.;

      for (unsigned int q=0; q<parallel_data.cell_quadrature.size(); ++q)
        parallel_data.diffusion_values_cell[q] = out.thermal_conductivities[q] / (out.densities[q] * out.specific_heat[q] + heating_model_outputs.lhs_latent_heat_terms[q]);

      std::vector<double> weight_values (n_q_points);
      std::vector<Tensor<1,dim,double> > grad_weight_values (n_q_points);
      std::vector<double> L (n_q_points);
      double              lambda_sq_cell = std::numeric_limits<double>::signaling_NaN();

      double weight_cell_max          = std::numeric_limits<double>::min();
      double weight_cell_min          = std::numeric_limits<double>::max();
      double grad_weight_sq_cell_max  = std::numeric_limits<double>::min();
      double L_cell_min               = std::numeric_limits<double>::max();
      double cell_weight_squared      = std::numeric_limits<double>::signaling_NaN();

      helmholtz_fe_values.reinit(helmholtz_cell);

      helmholtz_fe_values.get_present_fe_values()
      .get_function_values (parallel_data.helmholtz_solution, parallel_data.value_H);

      helmholtz_fe_values.get_present_fe_values()
      .get_function_gradients (parallel_data.helmholtz_solution, parallel_data.gradient_H);


      parallel_data.weighting_function->value_list (parallel_data.fe_values_cell.
                                                    get_present_fe_values().get_quadrature_points(),
                                                    parallel_data.value_H,
                                                    weight_values);
      parallel_data.weighting_function->gradient_list (parallel_data.fe_values_cell.
                                                       get_present_fe_values().get_quadrature_points(),
                                                       parallel_data.value_H,
                                                       parallel_data.gradient_H,
                                                       grad_weight_values);

      parallel_data.reaction_function->value_list (parallel_data.fe_values_cell.
                                                   get_present_fe_values().get_quadrature_points(),
                                                   parallel_data.div_u_cell,
                                                   parallel_data.diffusion_values_cell,
                                                   parallel_data.gradient_H,
                                                   parallel_data.velocity_values,
                                                   parallel_data.reaction_values);

      //loop over points, calculating L at each point.
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          L[q] = parallel_data.reaction_values[q] + 1./2. * (EquationData::alpha * parallel_data.gradient_H[q]*parallel_data.velocity_values[q]
                                                             -
                                                             (1-EquationData::alpha * parallel_data.diffusion_values_cell[q]) * parallel_data.div_u_cell[q]
                                                             -
                                                             EquationData::alpha * EquationData::alpha * parallel_data.diffusion_values_cell[q] *parallel_data.gradient_H[q] * parallel_data.gradient_H[q]);
          Assert (L[q] >= 0,
                  ExcMessage ("L(q) < 0"));
        }
      //find maxes and mins of each value as appropriate
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          weight_cell_max = std::max(weight_cell_max,
                                     weight_values[q]);
          weight_cell_min = std::min(weight_cell_min,
                                     weight_values[q]);
          grad_weight_sq_cell_max = std::max(grad_weight_sq_cell_max,
                                             grad_weight_values[q].norm());
          L_cell_min = std::min(L_cell_min,
                                L[q]);
        }

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          lambda_sq_cell = std::max(grad_weight_sq_cell_max/L_cell_min,
                                    weight_cell_max*weight_cell_max/parallel_data.diffusion_values_cell[q]);
        }

      //calculate cell weight squared
      cell_weight_squared = weight_cell_min*std::min(weight_cell_max*weight_cell_max/L_cell_min,
                                                     cell->diameter()*cell->diameter()*lambda_sq_cell);



      /* This section calculates the projection of the RHS function on the cell,
       * and then does NOT neeed to save to a global vector, as quadrature
       * points and dofs are ordered the same.
       */
      const unsigned int
      dofs_per_cell = projection_fe_values.dofs_per_cell;
      Vector<double>      local_projection (dofs_per_cell);
      {
        // set up necessary variables
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        Vector<double>      cell_vector (dofs_per_cell);
        FullMatrix<double>  local_mass_matrix (dofs_per_cell, dofs_per_cell);
        std::vector<double> heating_values(n_q_points);

        // initialise everything for this cell
        projection_cell->get_dof_indices (local_dof_indices);
        projection_fe_values.reinit(projection_cell);

        // get all data and pass to material and heating models
        MaterialModel::MaterialModelInputs<dim> material_model_inputs (n_q_points,this->get_parameters().n_compositional_fields);
        MaterialModel::MaterialModelOutputs<dim> material_model_outputs (n_q_points, this->get_parameters().n_compositional_fields);

        std::vector<Point<dim> > position = projection_fe_values.get_quadrature_points();
        material_model_inputs.position = position;

        parallel_data.fe_values_cell[this->introspection().extractors.temperature].get_function_values(this->get_solution(),
            material_model_inputs.temperature);
        parallel_data.fe_values_cell[this->introspection().extractors.pressure].get_function_values(this->get_solution(),
            material_model_inputs.pressure);
        parallel_data.fe_values_cell[this->introspection().extractors.velocities].get_function_values(this->get_solution(),
            material_model_inputs.velocity);
        parallel_data.fe_values_cell[this->introspection().extractors.pressure].get_function_gradients(this->get_solution(),
            material_model_inputs.pressure_gradient);

        // the values of the compositional fields are stored as blockvectors for each field
        // we have to extract them in this structure
        std::vector<std::vector<double> > composition_values (this->get_parameters().n_compositional_fields,
                                                              std::vector<double> (n_q_points));

        for (unsigned int c=0; c<this->get_parameters().n_compositional_fields; ++c)
          parallel_data.fe_values_cell[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(),
              composition_values[c]);

        // then we copy these values to exchange the inner and outer vector, because for the material
        // model we need a vector with values of all the compositional fields for every quadrature point
        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int c=0; c<this->get_parameters().n_compositional_fields; ++c)
            material_model_inputs.composition[q][c] = composition_values[c][q];

        material_model_inputs.cell = &projection_cell;

        this->get_material_model().evaluate(material_model_inputs,
                                            material_model_outputs);
        MaterialModel::MaterialAveraging::average (this->get_parameters().material_averaging,
                                                   projection_cell,
                                                   projection_quadrature,
                                                   projection_fe_values.get_mapping(),
                                                   material_model_outputs);

        HeatingModel::HeatingModelOutputs heating_model_outputs(n_q_points, this->get_parameters().n_compositional_fields);
        this->get_heating_model_manager().evaluate(material_model_inputs,
                                                   material_model_outputs,
                                                   heating_model_outputs);

        // evaluation done. set up for, and execute, assembly loop

        cell_vector = 0;
        local_mass_matrix = 0;
        for (unsigned int point=0; point<n_q_points; ++point)
          {
            const double gamma = heating_model_outputs.heating_source_terms[point] / (material_model_outputs.densities[point] * material_model_outputs.specific_heat[point] + heating_model_outputs.lhs_latent_heat_terms[point])
                                 + parallel_data.reaction_values[point] * parallel_data.temperature_values[point];

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                // populate the local matrix
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  local_mass_matrix(i,j) += (projection_fe_values[extract_projection].value(i,point) *
                                             projection_fe_values[extract_projection].value(j,point) *
                                             projection_fe_values.JxW(point));

                cell_vector(i) += gamma * //here gamma = heating + d*T
                                  projection_fe_values[extract_projection].value(i,point) *
                                  projection_fe_values.JxW(point);
              }
          }

        // now invert the local mass matrix and multiply it with the rhs
        local_mass_matrix.gauss_jordan();
        local_mass_matrix.vmult (local_projection, cell_vector);
      }



      double timestep = (this->get_timestep() == 0 ? 1 : this->get_timestep() );

      for (unsigned int q=0; q<parallel_data.cell_quadrature.size(); ++q)
        {
          velocity_grad_temperature = 0.;
          for (unsigned int component=0; component<dim; ++component)
            {
              velocity_grad_temperature += parallel_data.velocity_values[q][component]*parallel_data.temperature_gradients[q][component];
            }
//TOOD: add proj(delta*T)-delta*T
          Assert (numbers::is_finite(Utilities::fixed_power<2>(local_projection[q]
                                                               - ((parallel_data.temperature_values[q] - parallel_data.old_temperature_values[q])/timestep)
                                                               + parallel_data.diffusion_values_cell[q] * parallel_data.temperature_laplacians[q]
                                                               - velocity_grad_temperature
                                                               - parallel_data.reaction_values[q] * parallel_data.temperature_values[q])
                                     * parallel_data.fe_values_cell.JxW(q)
                                     * cell_weight_squared),
                  ExcMessage("Cell contribution is not finite"));
          Assert (Utilities::fixed_power<2>(local_projection[q]
                                            - ((parallel_data.temperature_values[q] - parallel_data.old_temperature_values[q])/timestep)
                                            + parallel_data.diffusion_values_cell[q] * parallel_data.temperature_laplacians[q]
                                            - velocity_grad_temperature
                                            - parallel_data.reaction_values[q] * parallel_data.temperature_values[q])
                  * parallel_data.fe_values_cell.JxW(q)
                  * cell_weight_squared >= 0,
                  ExcMessage("Cell contribution is negative"));
          //the argument holds the cell term number. No need to worry about cell number, as we are only talking about the one local cell.
          cell_integral[0] += Utilities::fixed_power<2>(local_projection[q]
                                                        - ((parallel_data.temperature_values[q] - parallel_data.old_temperature_values[q])/timestep)
                                                        + parallel_data.diffusion_values_cell[q] * parallel_data.temperature_laplacians[q]
                                                        - velocity_grad_temperature
                                                        - parallel_data.reaction_values[q] * parallel_data.temperature_values[q])
                              * parallel_data.fe_values_cell.JxW(q)
                              * cell_weight_squared;
        }

      local_cell_integrals[cell] = cell_integral;
    }

    template <int dim>
    void
    DGErrorEstimator<dim>::execute(Vector<float> &indicators) const
    {
      // Check that we are using discontinuous temperature discretization
      AssertThrow(this->get_parameters().use_discontinuous_temperature_discretization,
                  ExcMessage("dg error estimator cannot be used unless "
                             "use_discontinuous_temperature_discretization is set to true."));

      /* Handle Ak:
       *    - Calculate old temperature projection - we in fact do nothing here, because we can't easily compute the projection from the old mesh - all our solution, old_solution etc have already been interpolated onto the new mesh. Instead, we just use old_solution as this shouldn't be too bad.
       *    - Calculate force projection. This is handled inside the integrate_over_cell function, as the projection can be done locally. We need to create a new FE_DGQ to talk about this vector - we don't want to use the FESystem that contains Stokes, temperature and compositions. We then just pass into integrate_over_cell a reference to the global vector, and the FEValues, Quadrature and Cell derived from the new FE_DGQ.
       */

      /* We have one cell term:
       * min{h^2/diffusion, 1./inf{\conv}^2_cell} ||Ak + diffusion laplacian(temperature) - conv \cdot grad(temperature)||^2
       */
      //TODO: add data terms for conv and f?
      //FUTURE: convert inf{\conv}^2_cell to over the patch of cell sharing a vertex with cell.
      const unsigned int n_cell_terms = 1;

      /* We have two face terms: min{diffusion*h, 1./inf{\conv}^2_face} ||[grad(temperature)]||^2
       * + max{stuff to be decided: currently 1} *
       * ( penalty*diffusion/h + max{conv}^2_face*h/diffusion} + max{conv}^2_face*diffusion/h
       * ||[temperature]||^2
       */
      //TODO: implement max{stuff to be decided}
      //FUTURE: convert first max{conv}^2_face to min{inftynorm{conv}^2_vertpatcharoundK, inftynorm{conv}^2_vertpatcharoundK'}
      const unsigned int n_face_terms = 2;

      FE_Q<dim> helmholtz_fe (this->get_parameters().stokes_velocity_degree);
      DoFHandler<dim> helmholtz_dof_handler (this->get_triangulation());
      helmholtz_dof_handler.distribute_dofs(helmholtz_fe);
      IndexSet helmholtz_locally_owned = helmholtz_dof_handler.locally_owned_dofs();
      IndexSet helmholtz_locally_relevant;
      DoFTools::extract_locally_relevant_dofs(helmholtz_dof_handler,
                                              helmholtz_locally_relevant);

      UpdateFlags helmholtz_update_flags = UpdateFlags(update_values   |
                                                       update_gradients   |
                                                       update_quadrature_points |
                                                       update_JxW_values);

      QGauss<dim> helmholtz_quadrature(this->get_parameters().stokes_velocity_degree+1);
      FEValues<dim> helmholtz_fe_values (this->get_mapping(), helmholtz_fe, helmholtz_quadrature, helmholtz_update_flags);

      LinearAlgebra::Vector helmholtz_solution;
      helmholtz_solution.reinit(helmholtz_locally_relevant);

      compute_helmholtz_decomposition(helmholtz_fe,
                                      helmholtz_dof_handler,
                                      helmholtz_update_flags,
                                      helmholtz_quadrature,
                                      helmholtz_fe_values,
                                      helmholtz_solution);

      EquationData::WeightingFunction<dim,double> weighting_function;
      const EquationData::WeightingFunction<dim,double>   *weighting_function_pointer;
      weighting_function_pointer = &weighting_function;

      EquationData::ReactionFunction<dim,double> reaction_function;
      const EquationData::ReactionFunction<dim,double>   *reaction_function_pointer;
      reaction_function_pointer = &reaction_function;

      DGParallelData<DoFHandler<dim> > parallel_data (this->get_fe(),
                                                      QGauss<dim> (this->get_parameters().temperature_degree+1),
                                                      QGauss<dim-1> (this->get_parameters().temperature_degree),
                                                      helmholtz_fe,
                                                      helmholtz_quadrature,
                                                      this->get_mapping(),
                                                      true,//need_quadrature_points,
                                                      this->get_timestep(),
                                                      n_cell_terms,
                                                      n_face_terms,
                                                      helmholtz_solution,
                                                      weighting_function_pointer,
                                                      reaction_function_pointer);
      indicators = 0;

      /* Set up FE_DGQ and DoFHandler for the projection of the RHS function, and
       * use these to set up projection_fe_values.
       */
      FE_DGQ<dim> projection_fe (this->get_parameters().temperature_degree);
      DoFHandler<dim> projection_dof_handler (this->get_triangulation());
      projection_dof_handler.distribute_dofs(projection_fe);
      IndexSet projection_locally_owned = projection_dof_handler.locally_owned_dofs();

      UpdateFlags projection_update_flags = UpdateFlags(update_values   |
                                                        update_quadrature_points |
                                                        update_JxW_values);

      QGauss<dim> projection_quadrature(this->get_parameters().temperature_degree+1);
      FEValues<dim> projection_fe_values (this->get_mapping(), projection_fe, projection_quadrature, projection_update_flags);

      //Setup global projection vector - not needed as the order of quadrature points is the same as that of dofs
      //      LinearAlgebra::Vector rhs_projection;
      //      rhs_projection.reinit (projection_locally_owned, this->get_mpi_communicator());

      // #terms, #cells. errors holds each of the terms for each of the cells,
      // before being combined at the final stage into indicators
      std::vector<Vector<float> >  errors;
      errors.resize(n_cell_terms+n_face_terms);
      for (unsigned int j=0; j<n_cell_terms+n_face_terms; ++j)
        errors[j].reinit(indicators.size()); //for each term, create a Vector of size indicators.size() i.e. #cells

      SampleLocalIntegrals<DoFHandler<dim> > local_integrals;
      // empty our own copy of the local face integrals and the local cell integrals for each solution vector
      local_integrals.local_face_integrals.clear();
      local_integrals.local_cell_integrals.clear();

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();
      typename DoFHandler<dim>::active_cell_iterator
      projection_cell = projection_dof_handler.begin_active(),
      projection_endc = projection_dof_handler.end();
      typename DoFHandler<dim>::active_cell_iterator
      helmholtz_cell = helmholtz_dof_handler.begin_active(),
      helmholtz_endc = helmholtz_dof_handler.end();

      // Loop over all locally-owned cells AND all ghost cells (neighbors of locally owned cells),
      // as we may need some ghost cells to calculate face terms.
      for (; ((cell!=endc) && (projection_cell!=projection_endc) && (helmholtz_cell!=helmholtz_endc)); ++cell, ++projection_cell, ++helmholtz_cell)
        if ((cell->is_locally_owned()) || (cell->is_ghost()))
          {
            parallel_data.resize();

            //TODO: place projection variables into a parallel data structure, or split cell PD and face PD?
            if (cell->is_locally_owned())
              integrate_over_cell(local_integrals.local_cell_integrals,
                                  cell,
                                  parallel_data,
                                  projection_cell,
                                  projection_fe_values,
                                  projection_quadrature,
                                  helmholtz_fe_values,
                                  helmholtz_cell);
            else
              local_integrals.local_cell_integrals[cell] = std::vector<float> (n_cell_terms);


            //regardless of whether we want the internal cell contribution, we may want a face associated with it:
            // loop over all faces of this cell
            for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
              {
                typename DoFHandler<dim>::face_iterator face = cell->face (face_no);

                // make sure we do work only once: this face may either be regular
                // or irregular. if it is regular and has a neighbor, then we visit
                // the face twice, once from every side. let the one with the lower
                // index do the work. if it is at the boundary, or if the face is
                // irregular, then do the work below
                if ((face->has_children() == false) &&
                    !cell->at_boundary(face_no) &&
                    (!cell->neighbor_is_coarser(face_no) &&
                     (cell->neighbor(face_no)->index() < cell->index() ||
                      (cell->neighbor(face_no)->index() == cell->index() &&
                       cell->neighbor(face_no)->level() < cell->level()))))
                  continue;

                // if the neighboring cell is less refined than the present one,
                // then do nothing since we integrate over the subfaces when we
                // visit the coarse cells.
                if (face->at_boundary() == false)
                  if (cell->neighbor_is_coarser(face_no))
                    continue;

                // if this face is part of the boundary but not of the neumann
                // boundary -> nothing to do. However, to make things easier when
                // summing up the contributions of the faces of cells, we enter this
                // face into the list of faces with contribution zero.

                //This isn't handled here because we DO care about both Neumann and Dirichlet boundaries
                //                if (face->at_boundary()
                //                    &&
                //                    (this->get_parameters().fixed_temperature_boundary_indicators.find(
                //#if DEAL_II_VERSION_GTE(8,3,0)
                //                       cell->face(face_no)->boundary_id()
                //#else
                //                       cell->face(face_no)->boundary_indicator()
                //#endif
                //                     )
                //                     == this->get_parameters().fixed_temperature_boundary_indicators.end()))
                //                  {
                //                    local_integrals.local_face_integrals[face]
                //                      = std::vector<float> (n_face_terms, 0.);
                //                    continue;
                //                  }

                // finally: note that we only have to do something if either the
                // present cell is on the subdomain we care for (and the same for
                // material_id), or if one of the neighbors behind the face is on
                // the subdomain we care for
                if ( ! ( cell->is_locally_owned()) )
                  {
                    // ok, cell is unwanted, but maybe its neighbor behind the face
                    // we presently work on? oh is there a face at all?
                    if (face->at_boundary())
                      continue;

                    bool care_for_cell = false;
                    if (face->has_children() == false)
                      care_for_cell |= (cell->neighbor(face_no)->is_locally_owned());
                    else
                      {
                        for (unsigned int sf=0; sf<face->n_children(); ++sf)
                          if (cell->neighbor_child_on_subface(face_no,sf)->is_locally_owned())
                            {
                              care_for_cell = true;
                              break;
                            }
                      }

                    // so if none of the neighbors cares for this subdomain or
                    // material either, then try next face
                    if (care_for_cell == false)
                      continue;
                  }

                // so now we know that we care for this face, let's do something
                // about it. first re-size the arrays we may use to the correct
                // size:
                parallel_data.resize();


                // then do the actual integration
                if (face->has_children() == false)
                  // if the face is a regular one, i.e.  either on the other side
                  // there is nirvana (face is at boundary), or the other side's
                  // refinement level is the same as that of this side, then handle
                  // the integration of these both cases together
                  integrate_over_regular_face (local_integrals.local_face_integrals,
                                               cell, face_no,
                                               helmholtz_cell,
                                               parallel_data,
                                               parallel_data.fe_face_values_cell,
                                               parallel_data.fe_face_values_neighbor);

                else
                  // otherwise we need to do some special computations which do not
                  // fit into the framework of the above function
                  integrate_over_irregular_face (local_integrals.local_face_integrals,
                                                 cell, face_no,
                                                 helmholtz_cell,
                                                 parallel_data,
                                                 parallel_data.fe_face_values_cell,
                                                 parallel_data.fe_subface_values);
              } //end for faces
          } //end if cell is locally owned or ghost



      Assert (((cell==endc) && (projection_cell==projection_endc)),
              ExcMessage("projection_cell or cell have not reached their end marker but the loop "
                         "over cells has ended. Investigate why they do not match.") );


      // now walk over all cells and collect information from the faces. only do
      // something if this is a cell we care for based on the subdomain id
      unsigned int present_cell=0;
      for (typename DoFHandler<dim>::active_cell_iterator cell=this->get_dof_handler().begin_active();
           cell!=this->get_dof_handler().end();
           ++cell, ++present_cell)
        if (cell->is_locally_owned())
          {
            for (unsigned int m=0; m<n_cell_terms; ++m)
              {
                errors[m](present_cell)
                += (local_integrals.local_cell_integrals[cell][m]);
              }

            // loop over all faces of this cell
            for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
              {
                Assert(local_integrals.local_face_integrals.find(cell->face(face_no))
                       != local_integrals.local_face_integrals.end(),
                       ExcInternalError());

                for (unsigned int m=0; m<n_face_terms; ++m)
                  {
                    // make sure that we have written a meaningful value into this
                    // slot
                    Assert (local_integrals.local_face_integrals[cell->face(face_no)][m] >= 0,
                            ExcInternalError());

                    errors[m+n_cell_terms](present_cell)
                    += (local_integrals.local_face_integrals[cell->face(face_no)][m]);
                  }
              }

            for (unsigned int j=0; j<n_cell_terms+n_face_terms; ++j)
              {
                indicators[present_cell] += errors[j](present_cell);
              }

            indicators[present_cell] = std::sqrt(indicators[present_cell]);
          }

    }


    template <int dim>
    void DGErrorEstimator<dim>::compute_helmholtz_decomposition(const FE_Q<dim> &helmholtz_fe,
                                                                const DoFHandler<dim> &helmholtz_dof_handler,
                                                                const UpdateFlags &helmholtz_update_flags,
                                                                const QGauss<dim> &helmholtz_quadrature,
                                                                FEValues<dim> &helmholtz_fe_values,
                                                                LinearAlgebra::Vector &helmholtz_solution) const
    {
      // TODO: should we use the extrapolated solution?

      //stuff for iterating over the mesh
      FEValues<dim> fe_values (this->get_mapping(), this->get_fe(), helmholtz_quadrature, helmholtz_update_flags);
      const unsigned int n_q_points = fe_values.n_quadrature_points;

      IndexSet helmholtz_locally_relevant_dofs (helmholtz_dof_handler.n_dofs());
      DoFTools::extract_locally_relevant_dofs (helmholtz_dof_handler,
                                               helmholtz_locally_relevant_dofs);
      //stuff for assembling system
      unsigned int dofs_per_cell = helmholtz_fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> cell_dof_indices (dofs_per_cell);
      Vector<double> cell_vector (dofs_per_cell);
      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);

      //stuff for getting the divergence values
      std::vector<double> velocity_values(helmholtz_fe_values.n_quadrature_points);

      //set up constraints
      ConstraintMatrix mass_matrix_constraints(helmholtz_locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(helmholtz_dof_handler, mass_matrix_constraints);

      VectorTools::interpolate_boundary_values (helmholtz_dof_handler,
                                                0,
                                                ZeroFunction<dim>(),
                                                mass_matrix_constraints);
      mass_matrix_constraints.close();

      //set up the matrix
      LinearAlgebra::SparseMatrix mass_matrix;
#ifdef ASPECT_USE_PETSC
      LinearAlgebra::DynamicSparsityPattern sp(helmholtz_dof_handler.locally_relevant_dofs());

#else
      TrilinosWrappers::SparsityPattern sp (helmholtz_dof_handler.locally_owned_dofs(),
                                            helmholtz_dof_handler.locally_owned_dofs(),
                                            helmholtz_locally_relevant_dofs,
                                            this->get_mpi_communicator());
#endif
      DoFTools::make_sparsity_pattern (helmholtz_dof_handler, sp, mass_matrix_constraints, false,
                                       Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
#ifdef ASPECT_USE_PETSC
      SparsityTools::distribute_sparsity_pattern(sp,
                                                 helmholtz_dof_handler.n_locally_owned_dofs_per_processor(),
                                                 this->get_mpi_communicator(), helmholtz_dof_handler.locally_relevant_dofs());

      sp.compress();
      mass_matrix.reinit (helmholtz_dof_handler.locally_owned_dofs(), helmholtz_dof_handler.locally_owned_dofs(), sp, this->get_mpi_communicator());
#else
      sp.compress();
      mass_matrix.reinit (sp);
#endif

      FEValuesExtractors::Vector extract_vel(0);

      //make distributed vectors.
      LinearAlgebra::Vector rhs, dist_solution;
      rhs.reinit(helmholtz_dof_handler.locally_owned_dofs(), this->get_mpi_communicator());
      dist_solution.reinit(helmholtz_dof_handler.locally_owned_dofs(), this->get_mpi_communicator());

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(), endc= this->get_dof_handler().end();
      typename DoFHandler<dim>::active_cell_iterator
      hcell = helmholtz_dof_handler.begin_active();

      for (; cell!=endc; ++cell, ++hcell)
        if (cell->is_locally_owned())
          {
            hcell->get_dof_indices (cell_dof_indices);
            helmholtz_fe_values.reinit (hcell);
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_divergences(this->get_solution(), velocity_values);

            cell_vector = 0;
            cell_matrix = 0;
            double divergence_scaling = 1;

            for (unsigned int point=0; point<n_q_points; ++point)
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                      cell_matrix(i,j) += helmholtz_fe_values.shape_grad(i,point)
                                          * helmholtz_fe_values.shape_grad(j,point)
                                          * helmholtz_fe_values.JxW(point);
                    }

                  cell_vector(i) += divergence_scaling * velocity_values[point] *
                                    helmholtz_fe_values.shape_value(i,point) *
                                    helmholtz_fe_values.JxW(point);
                }

            mass_matrix_constraints.distribute_local_to_global (cell_matrix, cell_vector,
                                                                cell_dof_indices, mass_matrix, rhs, false);
          }

      rhs.compress (VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);

      //Jacobi seems to be fine here.  Other preconditioners (ILU, IC) run into troubles
      //because the matrrix is mostly empty, since we don't touch internal vertices.
      LinearAlgebra::PreconditionJacobi preconditioner_mass;
      preconditioner_mass.initialize(mass_matrix);

      SolverControl solver_control(5*rhs.size(), this->get_parameters().linear_stokes_solver_tolerance*rhs.l2_norm());
      SolverCG<LinearAlgebra::Vector> cg(solver_control);
      cg.solve (mass_matrix, dist_solution, rhs, preconditioner_mass);

      mass_matrix_constraints.distribute (dist_solution);
      helmholtz_solution = dist_solution;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(DGErrorEstimator,
                                              "dg error estimator",
                                              "A mesh refinement criterion that computes "
                                              "refinement indicators from a temperature field discretized "
                                              "by discontinuous Galerkin finite elements. This computes "
                                              "cell- and edge-based terms that are derived from a posteriori "
                                              "error analysis of the method.")
  }
}

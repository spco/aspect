/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

  This file is part of ASPECT.

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


#include <aspect/postprocess/visualization/velocity_divergence.h>
#include <aspect/simulator_access.h>



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      VelocityDivergence<dim>::
      VelocityDivergence ()
        :
        DataPostprocessorScalar<dim> ("velocity_divergence",
                                      update_gradients | update_q_points)
      {}



      template <int dim>
      void
      VelocityDivergence<dim>::
      compute_derived_quantities_vector (const std::vector<Vector<double> > &,
                                         const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                         const std::vector<std::vector<Tensor<2,dim> > > &,
                                         const std::vector<Point<dim> > &,
                                         const std::vector<Point<dim> > &,
                                         std::vector<Vector<double> >                    &computed_quantities) const
      {
        const unsigned int n_quadrature_points = duh.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
        Assert (duh[0].size() == this->introspection().n_components,          ExcInternalError());

          double max_div_u = 0.;
        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            // sum up diagonal elements of gradient vector
            double div_u=0.;
            for (unsigned int d=0; d<dim; ++d)
              div_u += duh[q][d][d];
              max_div_u = std::max(max_div_u,std::abs(div_u));
            // output div u
            //computed_quantities[q](0) = div_u;
          }
          for (unsigned int q=0; q<n_quadrature_points; ++q)
              computed_quantities[q](0) = max_div_u;

      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(VelocityDivergence,
                                                  "velocity divergence",
                                                  "A visualization output object that generates output "
                                                  "for the divergence of the velocity field, i.e., for the quantity "
                                                  "$\\sum_{i=1}^{\\mathrm{dim}}\\frac{\\partial{\\mathbf u}_i}{\\partial{x_i}}$ ")
    }
  }
}

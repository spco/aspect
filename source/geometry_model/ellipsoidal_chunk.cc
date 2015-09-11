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


#include <aspect/geometry_model/ellipsoidal_chunk.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <boost/lexical_cast.hpp>


/**
 * This geometry model implements an (3d) ellipsoidal_chunk geometry which can be non-coordinate parallel.
 * @author This plugin is a joined effort of Menno Fraters, D Sarah Stamps and Wolfgang Bangerth
 */




namespace aspect
{
  namespace GeometryModel
  {
    using namespace dealii;

    /**
     * the EllipsoidalChunkGeometry class
     */

    // constructor
    template <int dim>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::EllipsoidalChunkGeometry()
      :
      semi_major_axis_a (-1),
      eccentricity (-1),
      semi_minor_axis_b (-1),
      rot_para_to_para_angle (0),
      para_to_rect_angle (0),
      rotation_longitude (-1),
      rotation_latitude (-1),
      bottom_depth (-1)
    {}

    template <int dim>
    void
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::set_initial_values(double para_semi_major_axis_a,
                                                                        double para_eccentricity,
                                                                        double para_semi_minor_axis_b,
                                                                        double para_rot_para_to_para_angle,
                                                                        double para_para_to_rect_angle,
                                                                        double para_rotation_longitude,
                                                                        double para_rotation_latitude,
                                                                        double para_bottom_depth)
    {
      semi_major_axis_a = para_semi_major_axis_a;
      eccentricity = para_eccentricity;
      semi_minor_axis_b = para_semi_minor_axis_b;
      rot_para_to_para_angle = para_rot_para_to_para_angle;
      para_to_rect_angle = para_para_to_rect_angle;
      rotation_longitude = para_rotation_longitude;
      rotation_latitude = para_rotation_latitude;
      bottom_depth = para_bottom_depth;
    }

    template <int dim>
    Point<3>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::push_forward_ellipsoid(const Point<3> &phi_theta_d, const double semi_major_axis_a, const double eccentricity) const
    {
      const double phi   = phi_theta_d[0];
      const double theta = phi_theta_d[1];
      const double d     = phi_theta_d[2];

      const double R_bar = semi_major_axis_a / std::sqrt(1 - (eccentricity * eccentricity *
                                                              std::sin(theta) * std::sin(theta)));

      return Point<3> ((R_bar + d) * std::cos(phi) * std::cos(theta),
                       (R_bar + d) * std::sin(phi) * std::cos(theta),
                       ((1 - eccentricity * eccentricity) * R_bar + d) * std::sin(theta));
    }

    template <int dim>
    Point<3>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::pull_back_ellipsoid(const Point<3> &x, const double semi_major_axis_a, const double eccentricity) const
    {
      const double R    = semi_major_axis_a;
      const double b      = std::sqrt(R * R * (1 - eccentricity * eccentricity));
      const double ep     = std::sqrt((R * R - b * b) / (b * b));
      const double p      = std::sqrt(x(0) * x(0) + x(1) * x(1));
      const double th     = std::atan2(R * x(2), b * p);
      const double phi    = std::atan2(x(1), x(0));
      const double theta  = std::atan2(x(2) + ep * ep * b * std::pow(std::sin(th),3),
                                       (p - (eccentricity * eccentricity * R  * std::pow(std::cos(th),3))));
      const double R_bar = R / (std::sqrt(1 - eccentricity * eccentricity * std::sin(theta) * std::sin(theta)));
      const double R_plus_d = p / std::cos(theta);

      Point<3> phi_theta_d;
      phi_theta_d[0] = phi;

      phi_theta_d[1] = theta;
      phi_theta_d[2] = R_plus_d - R_bar;
      return phi_theta_d;
    }

    /**
     * TODO: These functions (pull back and push forward) should be changed that they always
     * take an return 3D points, because 2D points make no sense for an ellipsoid, even with
     * a 2D triangulation. To do this correctly we need to add the spacedim to the triangulation
     * in ASPECT. What is now presented is just a temporary fix to get acces to the pull back
     * function from outside. The push forward function can't be fixed in this way, because
     * it is used by a bind statement.
     */
    template <int dim>
    Point<3>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::pull_back(const Point<3> &space_point) const
    {
      AssertThrow (dim == 3,ExcMessage ("This can not be done with 2D points."));
      return pull_back_ellipsoid (space_point, semi_major_axis_a, eccentricity);

    }

    template <int dim>
    Point<2>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::pull_back(const Point<2> &space_point) const
    {
      AssertThrow (dim == 3,ExcMessage ("This can not be done with 2D points."));
      return space_point;

    }

    template <int dim>
    Point<3>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::push_forward(const Point<3> &chart_point) const
    {
      AssertThrow (dim == 3,ExcMessage ("This can not be done with 2D points."));
      return push_forward_ellipsoid (chart_point, semi_major_axis_a, eccentricity);
    }


    template<int dim>
    Point<dim>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::rotate_rectangle_to_parallelogram(const Point<dim> &x) const
    {
      // rotate all points back relative to the corresponding point on the line of corner 0 to 1.
      Point <dim> transformed,reference;
      reference[0] = x[0];
      reference[1] = rotation_latitude*numbers::PI/180;
      if (dim == 3)
        {
          reference[2] = x[2];
          transformed[2] = x[2];
        }
      transformed[0] = std::cos(-para_to_rect_angle) * (x[0] - reference[0]) - std::sin(-para_to_rect_angle) * (x[1] - reference[1]) + reference[0];
      transformed[1] = std::sin(-para_to_rect_angle) * (x[0] - reference[0]) + std::cos(-para_to_rect_angle) * (x[1] - reference[1]) + reference[1];
      return transformed;
    }

    template<int dim>
    Point<dim>
    EllipsoidalChunk<dim>::EllipsoidalChunkGeometry::rotate_parallelogram(const Point<dim> &x) const
    {
      // rotate all points back relative to the corresponding point on the line of corner 0 to 1.
      Point <dim> transformed,reference;
      reference[0] = rotation_longitude*numbers::PI/180;
      reference[1] = rotation_latitude*numbers::PI/180;
      if (dim == 3)
        {
          reference[2] = x[2];
          transformed[2] = x[2];
        }
      transformed[0] = std::cos(rot_para_to_para_angle) * (x[0] - reference[0]) - std::sin(-rot_para_to_para_angle) * (x[1] - reference[1]) + reference[0];
      transformed[1] = std::sin(rot_para_to_para_angle) * (x[0] - reference[0]) + std::cos(-rot_para_to_para_angle) * (x[1] - reference[1]) + reference[1];
      return transformed;
    }

    template <>
    void
    EllipsoidalChunk<3>::create_coarse_mesh(parallel::distributed::Triangulation<3> &coarse_grid) const
    {
      const int dim = 3;
      //std::cout << "semi_major_axis_a: " <<  manifold.semi_major_axis_a << ":" << manifold.get_semi_major_axis_a() << std::endl;
      const Point<3> corner_points[2] = { Point<3>(corners_rectangle[0][0]*numbers::PI/180,
                                                   corners_rectangle[0][1]*numbers::PI/180,
                                                   -bottom_depth),
                                          Point<3>(corners_rectangle[2][0]*numbers::PI/180,
                                                   corners_rectangle[2][1]*numbers::PI/180,
                                                   0)
                                        };
      std::vector<unsigned int> subdivisions(3);
      subdivisions[0] = EW_subdiv; // divide east-west
      subdivisions[1] = NS_subdiv; // divide north-south length
      subdivisions[2] = depth_subdiv; // divide depth
      GridGenerator::subdivided_hyper_rectangle (coarse_grid, subdivisions,
                                                 corner_points[0], corner_points[1],
                                                 true);
      if (para_to_rect_angle != 0)
        {
          // transform rectangle back to parallelogram
          GridTools::transform (std_cxx11::bind(&EllipsoidalChunkGeometry::rotate_rectangle_to_parallelogram,
                                                std_cxx11::cref(manifold),
                                                std_cxx11::_1),
                                coarse_grid);
        }
      if (rot_para_to_para_angle != 0)
        {
          // transform parallelogram back to rotated parallelogram
          GridTools::transform (std_cxx11::bind(&EllipsoidalChunkGeometry::rotate_parallelogram,
                                                std_cxx11::cref(manifold),
                                                std_cxx11::_1),
                                coarse_grid);
        }
      //transform to the ellipsoid surface
      GridTools::transform (std_cxx11::bind(&EllipsoidalChunk<3>::EllipsoidalChunkGeometry::push_forward,
                                            std_cxx11::cref(manifold),
                                            std_cxx11::_1),
                            coarse_grid);

      // set all boundary indicators. we want the edges to be curved
      // as well. for this, it matters in which order we call
      // set_all_boundary_indicators() -- we have to do it last for
      // the inner and outer boundary, which conveniently is what
      // happens in the following loop
      for (Triangulation<3>::active_cell_iterator cell =
             coarse_grid.begin_active(); cell != coarse_grid.end(); ++cell)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
#if DEAL_II_VERSION_GTE(8,3,0)
            cell->face(f)->set_all_boundary_ids(f);
#else
            cell->face(f)->set_all_boundary_indicators(f);
#endif

      // also attach the real manifold to slot 15. we won't use it
      // during regular operation, but we set manifold_ids for all
      // cells, faces and edges immediately before refinement and
      // clear it again afterwards
      coarse_grid.set_manifold (15, manifold);

      coarse_grid.signals.pre_refinement.connect (std_cxx11::bind (&set_manifold_ids,
                                                                   std_cxx11::ref(coarse_grid)));
      coarse_grid.signals.post_refinement.connect (std_cxx11::bind (&clear_manifold_ids,
                                                                    std_cxx11::ref(coarse_grid)));
    }

    template <int dim>
    void
    EllipsoidalChunk<dim>::create_coarse_mesh(parallel::distributed::Triangulation<dim> &coarse_grid) const
    {
      Assert(false, ExcNotImplemented());
    }

    template <int dim>
    std::set<types::boundary_id>
    EllipsoidalChunk<dim>::get_used_boundary_indicators() const
    {
      // boundary indicators are zero through 2*dim-1
      std::set<types::boundary_id> s;
      for (unsigned int i = 0; i < 2 * dim; ++i)
        s.insert(i);
      return s;
    }

    template <int dim>
    std::map<std::string,types::boundary_id>
    EllipsoidalChunk<dim>::
    get_symbolic_boundary_names_map () const
    {
      switch (dim)
        {
          case 2:
          {
            static const std::pair<std::string,types::boundary_id> mapping[]
              = { std::pair<std::string,types::boundary_id>("east",   0),
                  std::pair<std::string,types::boundary_id>("west",  1),
                  std::pair<std::string,types::boundary_id>("inner", 2),
                  std::pair<std::string,types::boundary_id>("outer",    3)
                };

            return std::map<std::string,types::boundary_id> (&mapping[0],
                                                             &mapping[sizeof(mapping)/sizeof(mapping[0])]);
          }

          case 3:
          {
            static const std::pair<std::string,types::boundary_id> mapping[]
              = { std::pair<std::string,types::boundary_id>("east",   0),
                  std::pair<std::string,types::boundary_id>("west",  1),
                  std::pair<std::string,types::boundary_id>("south",  2),
                  std::pair<std::string,types::boundary_id>("north",   3),
                  std::pair<std::string,types::boundary_id>("inner", 4),
                  std::pair<std::string,types::boundary_id>("outer",    5)
                };

            return std::map<std::string,types::boundary_id> (&mapping[0],
                                                             &mapping[sizeof(mapping)/sizeof(mapping[0])]);
          }
        }

      Assert (false, ExcNotImplemented());
      return std::map<std::string,types::boundary_id>();
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::depth(const Point<dim> &position) const
    {
      /**
       * the center of the earth lies in the origin, so the norm of the current position gives
       * the distance from the center of the earth to this position. If we substract this from
       * the radius at this position, we get the depth.
       */
      return std::max(std::min(get_radius(position) - position.norm(), maximal_depth()), 0.0);
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::maximal_depth() const
    {
      return bottom_depth;
    }



    template <int dim>
    void
    EllipsoidalChunk<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Ellipsoidal chunk");
        {
          prm.declare_entry("NE corner",
                            "",
                            Patterns::Anything(),
                            "Longitude:latitude in degrees of the North-East corner point of model region."
                            "The North-East direction is positive. If one of the three corners is not provided"
                            "the missing corner value will be calculated so all faces are parallel.");
          prm.declare_entry("NW corner",
                            "",
                            Patterns::Anything(),
                            "Longitude:latitude in degrees of the North-West corner point of model region. "
                            "The North-East direction is positive. If one of the three corners is not provided"
                            "the missing corner value will be calculated so all faces are parallel.");
          prm.declare_entry("SW corner",
                            "",
                            Patterns::Anything(),
                            "Longitude:latitude in degrees of the South-West corner point of model region. "
                            "The North-East direction is positive. If one of the three corners is not provided"
                            "the missing corner value will be calculated so all faces are parallel.");
          prm.declare_entry("SE corner",
                            "",
                            Patterns::Anything(),
                            "Longitude:latitude in degrees of the South-East corner point of model region. "
                            "The North-East direction is positive. If one of the three corners is not provided"
                            "the missing corner value will be calculated so all faces are parallel.");
          prm.declare_entry("Depth",
                            "500000.0",
                            Patterns::Double(0),
                            "Bottom depth of model region.");
          prm.declare_entry("Semi-major axis",
                            "6378137.0",
                            Patterns::Double(0),
                            "The semi-major axis (a) of an ellipsoid. This is the radius for a sphere (eccentricity=0). Default WGS84 semi-major axis.");
          prm.declare_entry("Eccentricity",
                            "8.1819190842622e-2",
                            Patterns::Double(0),
                            "Eccentricity of the ellipsoid. Zero is a perfect sphere, default (8.1819190842622e-2) is WGS84.");
          prm.declare_entry("East-West subdivisions",
                            "1",
                            Patterns::Integer(0),
                            "The number of subdivisions of the coarse (initial) mesh in the East-West direction.");
          prm.declare_entry("North-South subdivisions",
                            "1",
                            Patterns::Integer(0),
                            "The number of subdivisions of the coarse (initial) mesh in the North-South direction.");
          prm.declare_entry("Depth subdivisions",
                            "1",
                            Patterns::Integer(0),
                            "The number of subdivisions of the coarse (initial) mesh in depth.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    EllipsoidalChunk<dim>::parse_parameters(ParameterHandler &prm)
    {
      // only implemented for the 3d case
      AssertThrow (dim == 3, ExcMessage ("2D has not been implemented."));
      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Ellipsoidal chunk");
        {
          /**
           * Get latitude and longitudes defining region of interest from
           * the parameter file.
           */
          corners.resize(4);
          std::string NEcorner = prm.get("NE corner");
          std::string NWcorner = prm.get("NW corner");
          std::string SWcorner = prm.get("SW corner");
          std::string SEcorner = prm.get("SE corner");

          /**
           * make a list of what corners are present and check that there should be one or two corner missing,
           * otherwise throw an exception
           */
          std::vector<bool> present(4,true);
          unsigned int missing = 0;
          if (NEcorner == "")
            {
              missing++;
              present[0] = false;
            }
          if (NWcorner == "")
            {
              missing++;
              present[1] = false;
            }
          if (SWcorner == "")
            {
              missing++;
              present[2] = false;
            }
          if (SEcorner == "")
            {
              missing++;
              present[3] = false;
            }

          AssertThrow (missing != 0,
                       ExcMessage ("Only two or three of the four corners points should be provided."));
          AssertThrow (missing == 1 || missing == 2,
                       ExcMessage ("Please provide two or three corners points."));

          corners[0] = Utilities::string_to_double(Utilities::split_string_list(NEcorner,':'));
          corners[1] = Utilities::string_to_double(Utilities::split_string_list(NWcorner,':'));
          corners[2] = Utilities::string_to_double(Utilities::split_string_list(SWcorner,':'));
          corners[3] = Utilities::string_to_double(Utilities::split_string_list(SEcorner,':'));
          rotation_longitude = corners[0][0];
          rotation_latitude = corners[0][1];
          bottom_depth = prm.get_double("Depth");
          semi_major_axis_a = prm.get_double("Semi-major axis");
          eccentricity = prm.get_double("Eccentricity");
          semi_minor_axis_b=std::sqrt((1 - pow(eccentricity,2)) * pow(semi_major_axis_a,2));
          EW_subdiv = prm.get_double("East-West subdivisions");
          NS_subdiv = prm.get_double("North-South subdivisions");
          depth_subdiv = prm.get_double("Depth subdivisions");

          // Check whether the corners of the rectangle are really place correctly
          if (present[0] == true && present[1] == true)
            {
              AssertThrow (corners[0][0] <= corners[1][0],
                           ExcMessage ("The longitude of the NE corner (" + boost::lexical_cast<std::string>(corners[0][0]) + ") cannot be smaller then the longitude of the NW corner (" + boost::lexical_cast<std::string>(corners[1][0]) + ")."));
            }
          if (present[1] == true && present[3] == true)
            {
              AssertThrow (corners[0][1] >= corners[3][1],
                           ExcMessage ("The latitude of the NE (" + boost::lexical_cast<std::string>(corners[0][1]) + ") corner cannot be larger then the longitude of the SE corner (" + boost::lexical_cast<std::string>(corners[3][1]) + ")."));
            }
          if (present[2] == true && present[3] == true)
            {
              AssertThrow (corners[2][0] >= corners[3][0],
                           ExcMessage ("The longitude of the SW (" + boost::lexical_cast<std::string>(corners[2][0]) + ") corner cannot be larger then the longitude of the SE corner (" + boost::lexical_cast<std::string>(corners[3][0]) + ")."));
            }
          if (present[3] == true && present[1] == true)
            {
              AssertThrow (corners[2][1] <= corners[1][1],
                           ExcMessage ("The latitude of the SW corner (" + boost::lexical_cast<std::string>(corners[2][1]) + ") cannot be smaller then the longitude of the NW corner (" + boost::lexical_cast<std::string>(corners[1][1]) + ")."));
            }
          if (missing == 2)
            {
              AssertThrow (present[0] == true && present[2] == true || present[1] == true && present[3] == true,
                           ExcMessage ("Please provide to opposing corners."));
            }




          // If one or two of the corners is not provided, calculate it.
          if (missing == 1)
            {
              for (unsigned int i = 0; i <= 3; i++)
                {
                  // find the corner which is missing, if there is one
                  if (present[i] == false)
                    {
                      // find the location of this corner
                      // the values for the values is:
                      // [i] = [i+1] + ([i+3] - [i+2])
                      unsigned int ip = i + 1;
                      unsigned int ipp = i + 2;
                      unsigned int ippp = i + 3;

                      if (ip > 3)
                        ip -= 4;
                      if (ipp > 3)
                        ipp -= 4;
                      if (ippp > 3)
                        ippp -=4;

                      corners[i] = corners[ip];
                      corners[i][0] = corners[ip][0] + (corners[ippp][0] - corners[ipp][0]);
                      corners[i][1] = corners[ip][1] + (corners[ippp][1] - corners[ipp][1]);
                    }
                }


              /**
               * We assume that this is a rotated parallelogram. But we can only make a mesh of
               * coordinate rectangles. So we have to transform the points to a rectangle and record
               * all the steps, make the mesh and transform the mesh back to the original form.
               *
               * Check what rotation is required to rotate the rectangle coordinate parallel.
               * To be more precise, we check how much corners[0] need to be rotated to be on
               * the same longitude as corners[1].
               * Then apply this rotation of corners[0] the other points and store the result in a different variable.
               */
              corners_parallelogram.resize(4);

              double a = corners[0][0] - corners[1][0];
              double o = corners[0][1] - corners[1][1];
              rot_para_to_para_angle = std::atan(o/a);

              // now rotate arround corner 0
              corners_parallelogram[0] = corners[0];
              for (unsigned int i = 1; i <= 3; i++)
                {
                  corners_parallelogram[i] = corners[i];
                  corners_parallelogram[i][0] = std::cos(-rot_para_to_para_angle) * (corners[i][0] - corners[0][0]) - std::sin(-rot_para_to_para_angle) * (corners[i][1] - corners[0][1]) + corners[0][0];
                  corners_parallelogram[i][1] = std::sin(-rot_para_to_para_angle) * (corners[i][0] - corners[0][0]) + std::cos(-rot_para_to_para_angle) * (corners[i][1] - corners[0][1]) + corners[0][1];

                }

              /**
               * Check what rotation is required to make from the parallelogram a rectangle.
               * To me more precise, we check what rotation is required to get corners[0]
               * and corner[3] on the same latitude. We then apply this rotation on corners[3]
               * relative to corners[0] and with a negative angle to corners[2] relative to
               * corners[1]
               */
              o = corners_parallelogram[0][0] - corners_parallelogram[3][0];
              a = corners_parallelogram[0][1] - corners_parallelogram[3][1];
              para_to_rect_angle = std::atan(o/a);
              corners_rectangle.resize(4);

              // now rotate arround corner 0 and 1
              corners_rectangle[0] = corners_parallelogram[0];
              corners_rectangle[0] = corners_parallelogram[0];
              corners_rectangle[1] = corners_parallelogram[1];
              corners_rectangle[2] = corners_parallelogram[2];
              corners_rectangle[3] = corners_parallelogram[3];
              corners_rectangle[2][0] = std::cos(para_to_rect_angle) * (corners_parallelogram[2][0] - corners_parallelogram[1][0]) - std::sin(para_to_rect_angle) * (corners_parallelogram[2][1] - corners_parallelogram[1][1]) + corners_parallelogram[1][0];
              corners_rectangle[2][1] = std::sin(para_to_rect_angle) * (corners_parallelogram[2][0] - corners_parallelogram[1][0]) + std::cos(para_to_rect_angle) * (corners_parallelogram[2][1] - corners_parallelogram[1][1]) + corners_parallelogram[1][1];
              corners_rectangle[3][0] = std::cos(para_to_rect_angle) * (corners_parallelogram[3][0] - corners_parallelogram[0][0]) - std::sin(para_to_rect_angle) * (corners_parallelogram[3][1] - corners_parallelogram[0][1]) + corners_parallelogram[0][0];
              corners_rectangle[3][1] = std::sin(para_to_rect_angle) * (corners_parallelogram[3][0] - corners_parallelogram[0][0]) + std::cos(para_to_rect_angle) * (corners_parallelogram[3][1] - corners_parallelogram[0][1]) + corners_parallelogram[0][1];
            }
          else if (missing == 2)
            {
              // find the location of this corner
              if (present[0] == false)
                {
                  corners[0] = corners[1];
                  corners[2] = corners[3];
                  corners[0][0] = corners[3][0];
                  corners[2][0] = corners[1][0];
                }
              else
                {
                  corners[1] = corners[0];
                  corners[3] = corners[2];
                  corners[1][0] = corners[2][0];
                  corners[3][0] = corners[0][0];
                }
              corners_rectangle = corners;
            }
          westLongitude = corners[2][0];
          eastLongitude = corners[0][0];
          northLatitude = corners[0][1];
          southLatitude = corners[2][1];
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();


      /**
       * Construct manifold object Pointer to an object that describes the geometry.
       */
      manifold.set_initial_values(semi_major_axis_a,
                                  eccentricity,
                                  semi_minor_axis_b,
                                  rot_para_to_para_angle,
                                  para_to_rect_angle,
                                  rotation_longitude,
                                  rotation_latitude,
                                  bottom_depth);
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_radius(const Point<dim> &position) const
    {
      return semi_major_axis_a / (std::sqrt(1 - eccentricity * eccentricity * std::sin(position[1]) * std::sin(position[1])));
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_eccentricity() const
    {
      return eccentricity;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_semi_minor_axis_b() const
    {
      return semi_minor_axis_b;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_semi_major_axis_a() const
    {
      return semi_major_axis_a;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_rot_para_to_para_angle() const
    {
      return rot_para_to_para_angle;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_para_to_rect_angle() const
    {
      return para_to_rect_angle;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_rotation_longitude() const
    {
      return rotation_longitude;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::get_rotation_latitude() const
    {
      return rotation_latitude;
    }

    template <int dim>
    double
    EllipsoidalChunk<dim>::length_scale() const
    {
      // diameter divided by 20
      return ((manifold.push_forward(Point<3>(southLatitude,
                                              eastLongitude,
                                              -bottom_depth))//,Radius,eccentricity)
               - manifold.push_forward(Point<3>(northLatitude, westLongitude, 0/*,Radius,eccentricity*/))).norm()
              / 20);
    }

    template <int dim>
    Point<dim>
    EllipsoidalChunk<dim>::representative_point(const double depth) const
    {
      return Point<dim>();
    }

    template <>
    Point<3>
    EllipsoidalChunk<3>::representative_point(const double depth) const
    {
      const int dim = 3;
      Assert(depth >= 0, ExcMessage("Given depth must be positive or zero."));
      Assert(depth <= maximal_depth(),
             ExcMessage("Given depth must be less than or equal to the maximal depth of this geometry."));

      // choose a point on the center axis of the domain
      Point<dim> p =
        (manifold.push_forward(Point<3>(southLatitude,
                                        eastLongitude,
                                        -bottom_depth))
         + manifold.push_forward(Point<3>(northLatitude, eastLongitude, 0)))
        / 2;
      p /= p.norm();
      p *= get_radius(p) - depth;
      return p;
    }
  }
}

template <int dim>
typename aspect::GeometryModel::EllipsoidalChunk<dim>::EllipsoidalChunkGeometry
aspect::GeometryModel::EllipsoidalChunk<dim>::get_manifold() const
{
  return manifold;
}

// explicit instantiations
namespace aspect
{
  namespace GeometryModel
  {
    ASPECT_REGISTER_GEOMETRY_MODEL(EllipsoidalChunk,
                                   "ellipsoidal chunk",
                                   "A 3D chunk geometry that accounts for Earth's ellipticity (default assuming the "
                                   "WGS84 ellipsoid definition) which can be defined in non-coordinate directions. "
                                   "In the description of the ellipsoidal chunk, two of the ellipsoidal axes have the "
                                   "same length so that there is only a semi-major axis and a semi-minor axis. "
                                   "The user has two options for creating an ellipsoidal chunk geometry: 1) by defining "
                                   "two opposing points (SW and NE or NW and SE) a coordinate parallel ellipsoidal "
                                   "chunk geometry will be created. 2) by defining three points a non-coordinate "
                                   "parallel ellipsoidal chunk will be created. The points are defined in the input "
                                   "file by longitude:latitude. It is also possible to define additional subdivisions of the "
                                   "mesh in each direction. Faces of the model are defined as 0, west; 1,east; 2, south; 3, "
                                   "north; 4, bottom; 5, top. ")
  }
}

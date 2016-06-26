#include <aspect/material_model/interface.h>
#include <aspect/velocity_boundary_conditions/interface.h>
#include <aspect/fluid_pressure_boundary_conditions/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/global.h>
#include <aspect/melt.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>


namespace aspect
{

  /**
    * A material model with constant material properties except for the
    * compaction viscosity, and non-zero melt and solid compressibilities.
    */
  template <int dim>
  class CompressibleMeltMaterial:
    public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
  {
    public:
      virtual bool is_compressible () const
      {
        return true;
      }

      virtual double reference_viscosity () const
      {
        return 1.5;
      }

      virtual double reference_density () const
      {
        return 1.0;
      }
      virtual void evaluate(const typename MaterialModel::Interface<dim>::MaterialModelInputs &in,
                            typename MaterialModel::Interface<dim>::MaterialModelOutputs &out) const
      {
        for (unsigned int i=0; i<in.position.size(); ++i)
          {
            out.viscosities[i] = 0.75;
            out.thermal_expansion_coefficients[i] = 0.0;
            out.specific_heat[i] = 1.0;
            out.thermal_conductivities[i] = 0.0;
            out.compressibilities[i] = 1.0;
            out.densities[i] = 1.0;
            for (unsigned int c=0; c<in.composition[i].size(); ++c)
              out.reaction_terms[i][c] = 0.0;
          }

        // fill melt outputs if they exist
        aspect::MaterialModel::MeltOutputs<dim> *melt_out = out.template get_additional_output<aspect::MaterialModel::MeltOutputs<dim> >();

        if (melt_out != NULL)
          {
            for (unsigned int i=0; i<in.position.size(); ++i)
              {
                melt_out->compaction_viscosities[i] = 0.5 * (in.position[i][1]+0.1)*(in.position[i][1]+0.1);
                melt_out->fluid_viscosities[i] = 1.0;
                melt_out->permeabilities[i] = 1.0;
                melt_out->fluid_density_gradients[i] = this->get_gravity_model().gravity_vector(in.position[i]);
                melt_out->fluid_densities[i] = 1.0;
              }
          }
      }
  };




  template <int dim>
  class RefFunction : public Function<dim>
  {
    public:
      RefFunction () : Function<dim>(2*dim+5) {}
      virtual void vector_value (const Point< dim >   &p,
                                 Vector< double >   &values) const
      {
        double x = p(0);
        double y = p(1);
        double porosity = -0.1 * std::exp(-(y+0.1))/(y+0.1) + 1.0;

        values[0]=0;       //x vel
        values[1]=0;    //y vel
        values[2]=0;  // p_f
        values[3]=0;  // p_c
        values[4]=0;       //x melt vel
        values[5]=0;    //y melt vel
        values[6]=0;  // p_s
        values[7]=0; // T
        values[8]=porosity; // porosity
      }
  };


  /**
    * A postprocessor that evaluates the accuracy of the solution
    * by using the L2 norm.
    */
  template <int dim>
  class CompressibleMeltPostprocessor : public Postprocess::Interface<dim>, public ::aspect::SimulatorAccess<dim>
  {
    public:
      /**
       * Generate graphical output from the current solution.
       */
      virtual
      std::pair<std::string,std::string>
      execute (TableHandler &statistics);
  };

  template <int dim>
  std::pair<std::string,std::string>
  CompressibleMeltPostprocessor<dim>::execute (TableHandler &statistics)
  {
    RefFunction<dim> ref_func;
    const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.velocities).degree+2);

    const unsigned int n_total_comp = this->introspection().n_components;
    Vector<float> cellwise_errors_porosity (this->get_triangulation().n_active_cells());
    ComponentSelectFunction<dim> comp_porosity(dim+2+dim+2, n_total_comp);

    VectorTools::integrate_difference (this->get_mapping(),this->get_dof_handler(),
                                       this->get_solution(),
                                       ref_func,
                                       cellwise_errors_porosity,
                                       quadrature_formula,
                                       VectorTools::L2_norm,
                                       &comp_porosity);

    double poro_l2 = std::sqrt(Utilities::MPI::sum(cellwise_errors_porosity.norm_sqr(),this->get_mpi_communicator()));

    std::ostringstream os;
    os << std::scientific << poro_l2;
    return std::make_pair("Error porosity_L2:", os.str());
  }

  template <int dim>
  class PressureBdry:

    public FluidPressureBoundaryConditions::Interface<dim>
  {
    public:
      virtual
      void fluid_pressure_gradient (
        const types::boundary_id boundary_indicator,
        const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
        const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
        const std::vector<Tensor<1,dim> > &normal_vectors,
        std::vector<double> &output
      ) const
      {
        for (unsigned int q=0; q<output.size(); ++q)
          {
            const double y = material_model_inputs.position[q][1];
            Tensor<1,dim> direction;
            direction[dim-1] = 1.0;
            output[q] = (1.1 + y)*direction * normal_vectors[q];
          }
      }



  };

}

// explicit instantiations
namespace aspect
{

  ASPECT_REGISTER_MATERIAL_MODEL(CompressibleMeltMaterial,
                                 "compressible melt material",
                                 "")


  ASPECT_REGISTER_POSTPROCESSOR(CompressibleMeltPostprocessor,
                                "compressible melt error",
                                "A postprocessor that compares the numerical solution to the analytical "
                                "solution derived for compressible melt transport in a 2D box as described "
                                "in the manuscript and reports the error.")

  ASPECT_REGISTER_FLUID_PRESSURE_BOUNDARY_CONDITIONS(PressureBdry,
                                                     "PressureBdry",
                                                     "A fluid pressure boundary condition that prescribes the "
                                                     "gradient of the fluid pressure at the boundaries as "
                                                     "calculated in the analytical solution. ")

}

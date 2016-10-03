//
//  vector_dependent_function.h
//  step-32
//
//  Created by Samuel Cox on 25/02/2015.
//  Copyright (c) 2015 Samuel Cox. All rights reserved.
//

#ifndef __step_32__vector_dependent_function__
#define __step_32__vector_dependent_function__

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
//#include <deal.II/base/std_cxx11/function.h>
#include <vector>
DEAL_II_NAMESPACE_OPEN

//template <typename number> class Vector;
//template <int rank, int dim, typename Number> class TensorFunction;
/**
 * This class is a model for a general function that, given a point at which
 * to evaluate the function, returns a vector of values with one or more
 * components.
 *
 * The class serves the purpose of representing both scalar and vector valued
 * functions. To this end, we consider scalar functions as a special case of
 * vector valued functions, in the former case only having a single component
 * return vector. Since handling vectors is comparatively expensive, the
 * interface of this class has functions which only ask for a single component
 * of the vector-valued results (this is what you will usually need in case
 * you know that your function is scalar-valued) as well as functions you can
 * ask for an entire vector of results with as many components as the function
 * object represents. Access to function objects therefore is through the
 * following methods:
 * @code
 * // access to one component at one point
 * double value (const Point<dim> &p,
 * const unsigned int component = 0) const;
 *
 * // return all components at one point
 * void vector_value (const Point<dim> &p,
 * Vector<double> &value) const;
 * @endcode
 *
 * For more efficiency, there are other functions returning one or all
 * components at a list of points at once:
 * @code
 * // access to one component at several points
 * void value_list (const std::vector<Point<dim> > &point_list,
 * std::vector<double> &value_list,
 * const unsigned int component = 0) const;
 *
 * // return all components at several points
 * void vector_value_list (const std::vector<Point<dim> > &point_list,
 * std::vector<Vector<double> > &value_list) const;
 * @endcode
 *
 * Furthermore, there are functions returning the gradient of the function or
 * even higher derivatives at one or several points.
 *
 * You will usually only overload those functions you need; the functions
 * returning several values at a time (value_list(), vector_value_list(), and
 * gradient analogs) will call those returning only one value (value(),
 * vector_value(), and gradient analogs), while those ones will throw an
 * exception when called but not overloaded.
 *
 * Conversely, the functions returning all components of the function at one
 * or several points (i.e. vector_value(), vector_value_list()), will
 * <em>not</em> call the function returning one component at one point
 * repeatedly, once for each point and component. The reason is efficiency:
 * this would amount to too many virtual function calls. If you have vector-
 * valued functions, you should therefore also provide overloads of the
 * virtual functions for all components at a time.
 *
 * Also note, that unless only called a very small number of times, you should
 * overload all sets of functions (returning only one value, as well as those
 * returning a whole array), since the cost of evaluation of a point value is
 * often less than the virtual function call itself.
 *
 * Support for time dependent functions can be found in the base class
 * FunctionTime.
 *
 *
 * <h3>Functions that return tensors</h3>
 *
 * If the functions you are dealing with have a number of components that are
 * a priori known (for example, <tt>dim</tt> elements), you might consider
 * using the TensorFunction class instead. This is, in particular, true if the
 * objects you return have the properties of a tensor, i.e., they are for
 * example dim-dimensional vectors or dim-by-dim matrices. On the other hand,
 * functions like VectorTools::interpolate or
 * VectorTools::interpolate_boundary_values definitely only want objects of
 * the current type. You can use the VectorFunctionFromTensorFunction class to
 * convert the former to the latter.
 *
 *
 * <h3>Functions that return different fields</h3>
 *
 * Most of the time, your functions will have the form $f : \Omega \rightarrow
 * {\mathbb R}^{n_\text{components}}$. However, there are occasions where you
 * want the function to return vectors (or scalars) over a different number
 * field, for example functions that return complex numbers or vectors of
 * complex numbers: $f : \Omega \rightarrow {\mathbb
 * C}^{n_\text{components}}$. In such cases, you can use the second template
 * argument of this class: it describes the scalar type to be used for each
 * component of your return values. It defaults to @p double, but in the
 * example above, it could be set to <code>std::complex@<double@></code>.
 *
 *
 * @ingroup functions
 * @author Wolfgang Bangerth, 1998, 1999, Luca Heltai 2014
 */
template <int dim, typename Number=double>
class CoefficientFunction:// : public CoefficientFunctionTime<Number>,
  public Subscriptor
{
  public:
    /**
     * Export the value of the template parameter as a static member constant.
     * Sometimes useful for some expression template programming.
     */
    static const unsigned int dimension = dim;
    /**
     * Number of vector components.
     */
    const unsigned int n_components;
    /**
     * Constructor. May take an initial value for the number of components
     * (which defaults to one, i.e. a scalar function), and the time variable,
     * which defaults to zero.
     */
    CoefficientFunction (const unsigned int n_components = 1,
                         const Number initial_time = 0.0);
    /**
     * Virtual destructor; absolutely necessary in this case.
     *
     * This destructor is declared pure virtual, such that objects of this class
     * cannot be created. Since all the other virtual functions have a pseudo-
     * implementation to avoid overhead in derived classes, they can not be
     * abstract. As a consequence, we could generate an object of this class
     * because none of this class's functions are abstract.
     *
     * We circumvent this problem by making the destructor of this class
     * abstract virtual. This ensures that at least one member function is
     * abstract, and consequently, no objects of type Function can be created.
     * However, there is no need for derived classes to explicitly implement a
     * destructor: every class has a destructor, either explicitly implemented
     * or implicitly generated by the compiler, and this resolves the
     * abstractness of any derived class even if they do not have an explicitly
     * declared destructor.
     *
     * Nonetheless, since derived classes want to call the destructor of a base
     * class, this destructor is implemented (despite it being pure virtual).
     */
    virtual ~CoefficientFunction () = 0;
    /**
     * Assignment operator. This is here only so that you can have objects of
     * derived classes in containers, or assign them otherwise. It will raise an
     * exception if the object from which you assign has a different number of
     * components than the one being assigned to.
     */
    CoefficientFunction &operator= (const CoefficientFunction &f);
    /**
     * Return the value of the function at the given point. Unless there is only
     * one component (i.e. the function is scalar), you should state the
     * component you want to have evaluated; it defaults to zero, i.e. the first
     * component.
     */
    virtual Number value (const Point<dim> &p,
                          const Number &t,
                          const unsigned int component = 0) const;
    virtual Number value (const Point<dim> &p,
                          const Number &t,
                          const Tensor<1,dim,Number> &tensor_1,
                          const Tensor<1,dim,Number> &tensor_2,
                          const unsigned int component = 0) const;
    virtual Number value (const Point<dim> &p,
                          const Number &t,
                          const Number &t_2,
                          const Tensor<1,dim,Number> &tensor_1,
                          const Tensor<1,dim,Number> &tensor_2,
                          const unsigned int component = 0) const;
    /**
     * Return all components of a vector-valued function at a given point.
     *
     * <tt>values</tt> shall have the right size beforehand, i.e. #n_components.
     *
     * The default implementation will call value() for each component.
     */
    virtual void vector_value (const Point<dim> &p,
                               const Number &t,
                               Vector<Number> &values) const;
    virtual void vector_value (const Point<dim> &p,
                               const Number &t,
                               const Tensor<1,dim,Number> &tensor_1,
                               const Tensor<1,dim,Number> &tensor_2,
                               Vector<Number> &values) const;
    virtual void vector_value (const Point<dim> &p,
                               const Number &t,
                               const Number &t_2,
                               const Tensor<1,dim,Number> &tensor_1,
                               const Tensor<1,dim,Number> &tensor_2,
                               Vector<Number> &values) const;
    /**
     * Set <tt>values</tt> to the point values of the specified component of the
     * function at the <tt>points</tt>. It is assumed that <tt>values</tt>
     * already has the right size, i.e. the same size as the <tt>points</tt>
     * array.
     *
     * By default, this function repeatedly calls value() for each point
     * separately, to fill the output array.
     */
    virtual void value_list (const std::vector<Point<dim> > &points,
                             const std::vector<Number> &temperatures,
                             std::vector<Number> &values,
                             const unsigned int component = 0) const;
    virtual void value_list (const std::vector<Point<dim> > &points,
                             const std::vector<Number> &temperatures,
                             const std::vector<Tensor<1,dim,Number> > &tensor_1,
                             const std::vector<Tensor<1,dim,Number> > &tensor_2,
                             std::vector<Number> &values,
                             const unsigned int component = 0) const;
    virtual void value_list (const std::vector<Point<dim> > &points,
                             const std::vector<Number> &temperatures,
                             const std::vector<Number> &temperatures_2,
                             const std::vector<Tensor<1,dim,Number> > &tensor_1,
                             const std::vector<Tensor<1,dim,Number> > &tensor_2,
                             std::vector<Number> &values,
                             const unsigned int component = 0) const;
    /**
     * Set <tt>values</tt> to the point values of the function at the
     * <tt>points</tt>. It is assumed that <tt>values</tt> already has the
     * right size, i.e. the same size as the <tt>points</tt> array, and that
     * all elements be vectors with the same number of components as this
     * function has.
     *
     * By default, this function repeatedly calls vector_value() for each point
     * separately, to fill the output array.
     */
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    const std::vector<Number> &temperatures,
                                    std::vector<Vector<Number> > &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    const std::vector<Number> &temperatures,
                                    const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                    const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                    std::vector<Vector<Number> > &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    const std::vector<Number> &temperatures,
                                    const std::vector<Number> &temperatures2,
                                    const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                    const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                    std::vector<Vector<Number> > &values) const;
    /**
     * For each component of the function, fill a vector of values, one for each
     * point.
     *
     * The default implementation of this function in Function calls
     * value_list() for each component. In order to improve performance, this
     * can be reimplemented in derived classes to speed up performance.
     */
    virtual void vector_values (const std::vector<Point<dim> > &points,
                                const std::vector<Number> &temperatures,
                                std::vector<std::vector<Number> > &values) const;
    virtual void vector_values (const std::vector<Point<dim> > &points,
                                const std::vector<Number> &temperatures,
                                const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                std::vector<std::vector<Number> > &values) const;
    virtual void vector_values (const std::vector<Point<dim> > &points,
                                const std::vector<Number> &temperatures,
                                const std::vector<Number> &temperatures2,
                                const std::vector<Tensor<1,dim,Number> > &tensor_1,
                                const std::vector<Tensor<1,dim,Number> > &tensor_2,
                                std::vector<std::vector<Number> > &values) const;
    /**
     * Return the gradient of the specified component of the function at the
     * given point.
     */
    virtual Tensor<1,dim,Number> gradient (const Point<dim> &p,
                                           const Number &t,
                                           const unsigned int component = 0) const;
    virtual Tensor<1,dim,Number> gradient (const Point<dim> &p,
                                           const Number &t,
                                           const Tensor<1,dim,Number> &gradt,
                                           const unsigned int component = 0) const;
    /**
     * Return the gradient of all components of the function at the given point.
     */
    virtual void vector_gradient (const Point<dim> &p,
                                  const Number &temperature,
                                  std::vector<Tensor<1,dim, Number> > &gradients) const;
    virtual void vector_gradient (const Point<dim> &p,
                                  const Number &temperature,
                                  const Tensor<1,dim,Number> &gradt,
                                  std::vector<Tensor<1,dim, Number> > &gradients) const;
    /**
     * Set <tt>gradients</tt> to the gradients of the specified component of the
     * function at the <tt>points</tt>. It is assumed that <tt>gradients</tt>
     * already has the right size, i.e. the same size as the <tt>points</tt>
     * array.
     */
    virtual void gradient_list (const std::vector<Point<dim> > &points,
                                const std::vector<Number> &temperatures,
                                std::vector<Tensor<1,dim, Number> > &gradients,
                                const unsigned int component = 0) const;
    virtual void gradient_list (const std::vector<Point<dim> > &points,
                                const std::vector<Number> &temperatures,
                                const std::vector<Tensor<1,dim,Number> > &gradts,
                                std::vector<Tensor<1,dim, Number> > &gradients,
                                const unsigned int component = 0) const;
    /**
     * For each component of the function, fill a vector of gradient values, one
     * for each point.
     *
     * The default implementation of this function in Function calls
     * value_list() for each component. In order to improve performance, this
     * can be reimplemented in derived classes to speed up performance.
     */
    virtual void vector_gradients (const std::vector<Point<dim> > &points,
                                   const std::vector<Number> &temperatures,
                                   std::vector<std::vector<Tensor<1,dim, Number> > > &gradients) const;
    virtual void vector_gradients (const std::vector<Point<dim> > &points,
                                   const std::vector<Number> &temperatures,
                                   const std::vector<Tensor<1,dim,Number> > &gradts,
                                   std::vector<std::vector<Tensor<1,dim, Number> > > &gradients) const;
    /**
     * Set <tt>gradients</tt> to the gradients of the function at the
     * <tt>points</tt>, for all components. It is assumed that
     * <tt>gradients</tt> already has the right size, i.e. the same size as the
     * <tt>points</tt> array.
     *
     * The outer loop over <tt>gradients</tt> is over the points in the list,
     * the inner loop over the different components of the function.
     */
    virtual void vector_gradient_list (const std::vector<Point<dim> > &points,
                                       const std::vector<Number> &temperatures,
                                       std::vector<std::vector<Tensor<1,dim, Number> > > &gradients) const;
    virtual void vector_gradient_list (const std::vector<Point<dim> > &points,
                                       const std::vector<Number> &temperatures,
                                       const std::vector<Tensor<1,dim,Number> > &gradts,
                                       std::vector<std::vector<Tensor<1,dim, Number> > > &gradients) const;
    /**
     * Compute the Laplacian of a given component at point <tt>p</tt>.
     */
    virtual Number laplacian (const Point<dim> &p,
                              const Number &t,
                              const unsigned int component = 0) const;
    /**
     * Compute the Laplacian of all components at point <tt>p</tt> and store
     * them in <tt>values</tt>.
     */
    virtual void vector_laplacian (const Point<dim> &p,
                                   const Number &t,
                                   Vector<Number> &values) const;
    /**
     * Compute the Laplacian of one component at a set of points.
     */
    virtual void laplacian_list (const std::vector<Point<dim> > &points,
                                 const std::vector<Number> &temperatures,
                                 std::vector<Number> &values,
                                 const unsigned int component = 0) const;
    /**
     * Compute the Laplacians of all components at a set of points.
     */
    virtual void vector_laplacian_list (const std::vector<Point<dim> > &points,
                                        const std::vector<Number> &temperatures,
                                        std::vector<Vector<Number> > &values) const;
    /**
     * Determine an estimate for the memory consumption (in bytes) of this
     * object. Since sometimes the size of objects can not be determined exactly
     * (for example: what is the memory consumption of an STL <tt>std::map</tt>
     * type with a certain number of elements?), this is only an estimate.
     * however often quite close to the true value.
     */
    std::size_t memory_consumption () const;

    /**
     * Return the value of the time variable/
     */
    Number get_time () const;
    /**
     * Set the time to <tt>new_time</tt>, overwriting the old value.
     */
    virtual void set_time (const Number new_time);
    /**
     * Advance the time by the given time step <tt>delta_t</tt>.
     */
    virtual void advance_time (const Number delta_t);

  private:
    /**
     * Store the present time.
     */
    Number time;

    /**
     * Exception
     */
    DeclException0 (ExcCoefficientFunctionGradientCalled);
};

template<int dim, typename Number>
inline Number
CoefficientFunction<dim, Number>::get_time () const
{
  return time;
}

DEAL_II_NAMESPACE_CLOSE

#endif /* defined(__step_32__vector_dependent_function__) */

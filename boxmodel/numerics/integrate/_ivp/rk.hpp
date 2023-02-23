#ifndef RK_HPP
#define RK_HPP

#include "base.hpp"

namespace BoxModel {
    class RungeKutta : public OdeSolver {
        public:
            RungeKutta(pybind11::function f, double t0, double tmax, pybind11::array_t<double> y0, double first_step):
                OdeSolver(f, t0, tmax, y0, first_step) 
            {
                this->f = fprime_wrapped(t0, this->y);
            }
            RungeKutta(pybind11::function f, double t0, double tmax, double y0, double first_step):
                OdeSolver(f, t0, tmax, y0, first_step) 
            {
                this->f = fprime_wrapped(t0, this->y);
            }
            RungeKutta(pybind11::function f, double t0, double tmax, pybind11::array_t<double>  y0,
                double max_step, double first_step, double min_step) : 
                OdeSolver(f, t0, tmax, y0, max_step, first_step, min_step) 
            {
                this->f = fprime_wrapped(t0, this->y);
            }
            RungeKutta(pybind11::function f, double t0, double tmax, double y0,
                double max_step, double first_step, double min_step) : 
                OdeSolver(f, t0, tmax, y0, max_step, first_step, min_step) 
            {
                this->f = fprime_wrapped(t0, this->y);
            }
            RungeKutta(pybind11::function f, double t0, double tmax, pybind11::array_t<double> y0, 
                double atol, double rtol, double max_step, double first_step, double min_step) :
                OdeSolver(f, t0, fmax, y0, atol, rtol, max_step, first_step, min_step)
            {
                this->f = fprime_wrapped(t0, this->y);
            }
            RungeKutta(pybind11::function f, double t0, double tmax, double y0, 
                double atol, double rtol, double max_step, double first_step, double min_step) :
                OdeSolver(f, t0, fmax, y0, atol, rtol, max_step, first_step, min_step)
            {
                this->f = fprime_wrapped(t0, this->y);
            }

        protected:
            void solver_step() {
                
            }

            std::valarray<double> f;

            int n_stages;
            int order;

            double **a;
            double *b;
            double *c;             
    };
    class AdaptiveRungeKutta : public RungeKutta {
        public:
            AdaptiveRungeKutta(pybind11::function f, double t0, double tmax, pybind11::array_t<double> y0, double )

        protected:
            void solver_step();

            std::valarray<double> f;

            int n_stages;
            int order;

            double **a;
            double *b;
            double *c;             
    };
}

#endif /* RK_HPP */
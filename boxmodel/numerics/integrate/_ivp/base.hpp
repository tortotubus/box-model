#ifndef BASE_HPP
#define BASE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace BoxModel {
    class OdeSolver {
        public:
            OdeSolver(pybind11::function f, double t0, double tmax, double y0, double atol, double hmax, double hmin);
            void step();
            double get_y();
            double get_t();
            bool is_complete();
            bool has_failed();

        protected:
            virtual void solver_step() {}
            pybind11::function fprime;
            double f(double t, double y);
            double t_start, t_old, t, t_end;
            double h_max, h, h_min;
            double y;
            double atol;
            bool complete, failure;
    };
}

#endif /* BASE_HPP */
#ifndef BASE_HPP
#define BASE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <valarray>

namespace BoxModel {
    class OdeSolver {
        public:
            OdeSolver(pybind11::function f, double t0, double tf, pybind11::array_t<double> y0, double first_step)
            {
                this->fprime = f;

                this->t_start = t0;
                this->t_old = t0;
                this->t = t0;
                this->t_end = tf;

                this->h = h;

                this->complete = 0;
                this->failure = 0;

                pybind11::buffer_info buffer = y0.request();

                if (buffer.ndim != 1) { throw std::runtime_error("Number of dimensions must be one"); }

                this->n = buffer.shape[0];
                this->y = std::valarray<double>(this->n);

                double* y_ptr = (double*)buffer.ptr;

                for (int i = 0; i < this->n; i++) 
                {
                    this->y[i] = y_ptr[i];
                }
            }
            OdeSolver(pybind11::function f, double t0, double tf, double y0, double first_step)
            {
                this->fprime = f;

                this->t_start = t0;
                this->t_old = t0;
                this->t = t0;
                this->t_end = tf;

                this->h = first_step;

                this->complete = 0;
                this->failure = 0;

                this->n = 1;
                this->y = std::valarray<double>(this->n);
                this->y[0] = y0;
            }
            OdeSolver(pybind11::function f, double t0, double tmax, pybind11::array_t<double> y0, 
                double max_step, double first_step, double min_step) : OdeSolver(f, t0, tmax, y0, first_step) 
            {
                this->h_min = min_step;
                this->h_max = max_step;
            }
            OdeSolver(pybind11::function f, double t0, double tmax, double y0,
                double max_step, double first_step, double min_step) : OdeSolver(f, t0, tmax, y0, first_step)
            {
                this->h_min = min_step;
                this->h_max = max_step;
            }
            OdeSolver(pybind11::function f, double t0, double tmax, pybind11::array_t<double> y0, double atol, double rtol, 
                double max_step, double first_step, double min_step) : OdeSolver(f, t0, tmax, y0, max_step, first_step, min_step)
            {
                this->atol = atol;
                this->rtol = rtol;
            }
            OdeSolver(pybind11::function f, double t0, double tmax, double y0, double atol, double rtol, 
                double max_step, double first_step, double min_step) : OdeSolver(f, t0, tmax, y0, max_step, first_step, min_step)
            {
                this->atol = atol;
                this->rtol = rtol;
            }

            void step();
            pybind11::array_t<double> get_y();
            double get_t();
            bool is_complete();
            bool has_failed();

        protected:
            virtual void solver_step() {}
            pybind11::function fprime;
            std::valarray<double> fprime_wrapped(double t, std::valarray<double> args);
            double t_start, t_old, t, t_end;
            double h_min, h_old, h, h_max;
            double atol, rtol;
            std::valarray<double> y;
            int n;
            bool complete, failure;
    };
}

#endif /* BASE_HPP */
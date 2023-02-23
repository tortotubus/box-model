#include "base.hpp"

void BoxModel::OdeSolver::step() 
{
    if (this->failure) { return; }
    else if (this->complete) { return; }
    else { solver_step(); }
}

pybind11::array_t<double> BoxModel::OdeSolver::get_y() 
{
    auto result = pybind11::array_t<double>(this->n);
    pybind11::buffer_info result_buffer = result.request();
    double* result_ptr = (double*)result_buffer.ptr;
    for (int i = 0; i < this->n; i++) { result_ptr[i] = this->y[i]; }
    return result;
}

double BoxModel::OdeSolver::get_t()
{
    return this->t;
}

bool BoxModel::OdeSolver::is_complete() 
{
    return this->complete;
}

bool BoxModel::OdeSolver::has_failed()
{
    return this->failure;
}

std::valarray<double> BoxModel::OdeSolver::fprime_wrapped(double t, std::valarray<double> args) 
{
    auto args_nparr = pybind11::array_t<double>(this->n);
    pybind11::buffer_info args_nparr_buffer = args_nparr.request();
    double* args_nparr_ptr = (double*)args_nparr_buffer.ptr;

    for (int i = 0; i < this->n; i++) { args_nparr_ptr[i] = args[i]; }

    pybind11::object result = this->fprime(t, args_nparr);
    pybind11::array_t<double> f_nparr = result.cast<pybind11::array_t<double>>();

    pybind11::buffer_info f_nparr_buffer = f_nparr.request();
    double* f_nparr_ptr = (double*)f_nparr_buffer.ptr;
    std::valarray<double> f(this->n);

    for (int i = 0; i < this->n; i++) { f[i] = f_nparr_ptr[i]; }

    return f;
}
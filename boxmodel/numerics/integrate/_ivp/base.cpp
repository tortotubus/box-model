#include "base.hpp"

BoxModel::OdeSolver::OdeSolver(pybind11::function f, double a, double b, double y0, double atol, double hmax, double hmin)
{
    this->fprime = f;

    this->t_start = a;
    this->t_old = a;
    this->t = a;
    this->t_end = b;

    this->y = y0;

    this->atol = atol;

    this->h_max = hmax;
    this->h = hmax;
    this->h_min = hmin;

    this->complete = 0;
    this->failure = 0;
}

void BoxModel::OdeSolver::step() 
{
    if (this->failure) { return; }
    else if (this->complete) { return; }
    else { solver_step(); }
}

double BoxModel::OdeSolver::get_y() 
{
    return this->y;
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

double BoxModel::OdeSolver::f(double t, double y) 
{
    pybind11::object result = this->fprime(t, y);
    return result.cast<double>();
}
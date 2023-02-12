#include <pybind11/pybind11.h>

#include "base.hpp"
#include "rkf45.hpp"

PYBIND11_MODULE(numerics, m) {
    m.doc() = "module doc";

    pybind11::class_<BoxModel::OdeSolver>(m, "OdeSolver")
        .def(pybind11::init<pybind11::function, double, double, double, double, double, double>());
        .def("step", &BoxModel::OdeSolver::step)
        .def("get_y", &BoxModel::OdeSolver::get_y)
        .def("get_t", &BoxModel::OdeSolver::get_t)
        .def("is_complete", &BoxModel::OdeSolver::is_complete)
        .def("has_failed", &BoxModel::OdeSolver::has_failed);

    
    //pybind11::class_<BoxModel::RKF45, BoxModel::OdeSolver>(m, "RKF45")
    //   .def(pybind11::init<pybind11::function, double, double, double, double, double, double>());
}

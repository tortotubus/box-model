ivp_src = [
  '_ivp/_ivp.cpp',
  '_ivp/base.cpp',
  '_ivp/base.hpp',
  '_ivp/rkf45.cpp',
  '_ivp/rkf45.hpp',
]

ivp = py3.extension_module('_ivp',
  ivp_src,
  include_directories: [
    incdir_pybind11,
    incdir_numpy,
    '_ivp'
  ],
  dependencies: [py3_dep],
  override_options: ['cython_language=cpp'],
  install: true,
  subdir: 'boxmodel/numerics/integrate'
)

py3.install_sources([
    '__init__.py',
    'solve.py'
  ],
  subdir: 'boxmodel/numerics/integrate'
)

#subdir('_ivp')
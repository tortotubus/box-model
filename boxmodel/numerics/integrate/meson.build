ivp_src = [
  '_ivp/numerics.cpp',
  '_ivp/base.cpp',
  '_ivp/base.hpp',
  '_ivp/rkf45.cpp',
  '_ivp/rkf45.hpp',
]

ivp = py3.extension_module('_ivp',
  ivp_src 
  include_directories: [
    incdir_numpy,
    '_ivp'
  ],
  dependencies: [py3_dep],
  override_options: ['cython_language=cpp'],
  install: true,
  subdir: 'boxmodel/integrate'
)

py3.install_sources([
    '__init__.py',
  ],
  subdir: 'boxmodel/integrate'
)

subdir('ivp')
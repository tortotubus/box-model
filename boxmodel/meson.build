

incdir_numpy = run_command(py3,
    [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
    ],
    check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

cc = meson.get_compiler('c')

incdir_pybind11 = run_command(py3,
    [
        '-c',
        'import pybind11; print(pybind11.get_include())'
    ],
    check: true
).stdout().strip()

inc_pybind11 = include_directories(incdir_pybind11)

python_sources = [
    '__init__.py',
    '__main__.py',
    'test_numerics.py',
    'test_numerics_vectorized.py',
    'run_box_model.py'
]

py3.install_sources(
    python_sources,
    subdir: 'boxmodel'
)

compilers = {
  'C': cc,
  'CPP': cpp,
}

subdir('model')
subdir('numerics')
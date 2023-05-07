project(
    'boxmodel',
    'c', 'cpp',
    version: '0.0.1',
    default_options: [
        'buildtype=debugoptimized',
        'c_std=c99',
        'cpp_std=c++17',
    ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.0')
    error('SciPy requires GCC >= 8.0')
  endif
endif

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()

subdir('boxmodel')
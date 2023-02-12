def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import pybind11

    config = Configuration('integrate', parent_package, top_path)
    
    ivp_src = ['_ivp/base.cpp',
               '_ivp/rkf45.cpp']

    ivp_headers = ['_ivp/base.hpp',
                   '_ivp/rkf45.hpp']
    
    ivp_dep = ['_ivp/_ivp.cpp'] + ivp_headers + ivp_src

    ext = config.add_extension('_ivp',
                sources=['_ivp/_ivp.cpp'] + ivp_src,
                depends=ivp_dep)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
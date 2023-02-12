def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import get_info
    from numpy.distutils.misc_util import Configuration
    config = Configuration('boxmodel',parent_package,top_path)
    config.add_subpackage('model')
    config.add_subpackage('numerics')

if __name__ == '__main__':
    from numpy.distutils import setup
    setup(**configuration(top_path='').todict())
#!/usr/bin/env python

import os
import sys
import subprocess
import textwrap
import warnings


if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

import builtins

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def check_submodules():
    """ verify that the submodules are checked out and clean
        use `git submodule update --init`; on failure
    """
    if not os.path.exists('.git'):
        return
    with open('.gitmodules') as f:
        for l in f:
            if 'path' in l:
                p = l.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule %s missing' % p)


    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: %s' % line)


def parse_setuppy_commands():

    args = sys.argv[1:]

    if not args:
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg')

    for command in good_commands:
        if command in args:
            return True

    if 'install' in args:
        print(textwrap.dedent(""))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent(""))
        return False

    bad_commands = dict(
        test="",
        upload="",
        upload_docs="",
        easy_install="",
        clean="",
        check="",
        register="",
        bdist_dumb="",
        bdist="",
        flake8="",
        build_sphinx="",
        )

    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                     'register', 'check', 'install_data', 'install_headers',
                     'install_lib', 'install_scripts', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)


    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    warnings.warn("Unrecognized setuptools command ('{}'), proceeding with "
                  "generating Cython sources and expanding templates".format(
                  ' '.join(sys.argv[1:])))
    return True

def check_setuppy_command():
    run_build = parse_setuppy_commands()
    if run_build:
        try:
            pkgname = 'numpy'
            import numpy
            pkgname = 'pybind11'
            import pybind11
        except ImportError as exc:  # We do not have our build deps installed
            print(textwrap.dedent(
                    """Error: '%s' must be installed before running the build.
                    """
                    % (pkgname,)))
            sys.exit(1)

    return run_build

def configuration(parent_package='', top_path=None):
    from numpy.distutils.system_info import get_info, NotFoundError
    from numpy.distutils.misc_util import Configuration


    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('boxmodel')

    return config


def setup_package():
    # In maintenance branch, change np_maxversion to N+3 if numpy is at N
    # Update here, in pyproject.toml, and in scipy/__init__.py
    # Rationale: SciPy builds without deprecation warnings with N; deprecations
    #            in N+1 will turn into errors in N+3
    # For Python versions, if releases is (e.g.) <=3.9.x, set bound to 3.10
    np_minversion = '1.19.5'
    np_maxversion = '9.9.99'
    python_minversion = '3.8'
    python_maxversion = '3.10'
    req_np = 'numpy>={},<{}'.format(np_minversion, np_maxversion)
    req_py = '>={},<{}'.format(python_minversion, python_maxversion)

    metadata = dict(
        name='boxmodel',
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = check_setuppy_command()

    # Disable OSX Accelerate, it has too old LAPACK
    os.environ['ACCELERATE'] = 'None'

    # This import is here because it needs to be done before importing setup()
    # from numpy.distutils, but after the MANIFEST removing and sdist import
    # higher up in this file.
    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup
        metadata['configuration'] = configuration



if __name__ == '__main__':
    setup_package()

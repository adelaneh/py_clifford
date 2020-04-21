import subprocess
import sys
import os

# check if pip is installed. If not, raise an ImportError
PIP_INSTALLED = True

try:
    import pip
except ImportError:
    PIP_INSTALLED = False

if not PIP_INSTALLED:
    raise ImportError('pip is not installed.')

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

# check if setuptools is installed. If not, install setuptools
# automatically using pip.
install_and_import('setuptools')

from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def build_extensions(self):
        import pkg_resources                                                            
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    not numpy_incl in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)

if __name__ == "__main__":

    # find packages to be included. exclude benchmarks.
    packages = setuptools.find_packages(exclude=["benchmarks", "benchmarks.custom_benchmarks"])

    with open('README.rst') as f:
        LONG_DESCRIPTION = f.read()

    setuptools.setup(
        name='py_clifford',
        version='0.1.0',
        description='Python library for simulation of 2-line visual discrimination task.',
        long_description=LONG_DESCRIPTION,
        url='http://www.columbia.edu/~aa4348/py_clifford',
        author='Adel Ardalan',
        author_email='adel.ardalan@gmail.com',
        license='BSD',
        classifiers=[
            'Development Status :: 1 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Education',
            'License :: OSI Approved :: BSD License',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',                            
            'Topic :: Scientific/Engineering',
            'Topic :: Utilities',
            'Topic :: Software Development :: Libraries',
        ],
        packages=packages,
        install_requires=[
            'numpy >= 1.7.0',
            'six',
	    'pandas',
            'matplotlib',
            'tensorflow == 1.14.0',
            'tqdm',
        ],
        setup_requires=[
            'numpy >= 1.7.0'                                                   
            'tensorflow == 1.14.0',
            'tqdm',
        ],
        include_package_data=True,
        zip_safe=False
    )

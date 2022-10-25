from setuptools import setup, find_packages

setup(
    name='optsmiles',
    version='0.0.1',
    python_requires='>=3.6',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    author='Vitor Pereira',
    author_email='vpereira@ceb.uminho.pt',
    description='',
    license='Apache License Version 2.0',
    keywords='',
    url='',
    test_suite='tests',
)

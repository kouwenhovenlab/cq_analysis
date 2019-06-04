from setuptools import setup, find_packages

setup(
    name='cq_analysis',
    version='0.0.1',
    description='Cq analysis toolbox',
    author='Wolfgang Pfaff',
    author_email='wolfgangpfff@gmail.com',
    url='https://github.com/kouwenhovenlab/cq_analysis',
    packages=find_packages(),
    install_requires=[
        'pandas>=0.22',
        'xarray',
        'matplotlib',
        'numpy',
        'lmfit',
        'scipy',
        'holoviews',
        'qcodes',
    ],
)

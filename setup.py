from setuptools import setup, find_packages

setup(
    name='ihudeliver10_2d3d',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/coolteemf/IHUdeLiver10-2D_3D-deformable-registration.git',
    author='François Lecomte',
    author_email='francois.lecomte@inria.fr',
    description='This package contains code to generate'
                'projections to perform 2D-3D deformable registration '
                'using the IHU deLiver10 dataset.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'nibabel',
        'deepdrr',
        'scikit-image'
        'matplotlib'
        'pynrrd'
        'scipy'
        'torch'
        'json'
    ]
)
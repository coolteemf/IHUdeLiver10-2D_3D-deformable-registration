from setuptools import setup, find_packages

setup(
    name='ihudeliver10_2d3d',
    version='1.0.0',
    python_requires='==3.10.*',
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
        'numpy==1.23.5',
        'nibabel==5.2.1',
        'cupy-cuda12x==13.0.0',
        'cuda-python==12.3.0',
        'killeengeo@git+https://github.com/coolteemf/killeengeo.git#geom_xflow',
        'deepdrr@git+https://github.com/coolteemf/deepdrr.git',
        'scikit-image==0.19.3',
        'matplotlib==3.8.3',
        'pynrrd==1.0.0',
        'scipy==1.12.0',
        'torch==2.2.1',
    ]
)
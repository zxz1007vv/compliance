from setuptools import find_packages, setup

setup(
    name='wbc-compliance',
    version='1.0.0',
    author='Tifanny Portela',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Whole-body compliance environments and reinforcement-learning tools.',
    install_requires=['jaynes==0.9.2',
                      'params-proto==2.10.9',
                      'gym',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.23.5',
                      'tensorboard==2.14.0',
                      'moviepy==1.0.3',
                      'imageio'
                      ],
    extras_require={
        'wandb': ['wandb==0.15.0', 'wandb_osh'],
    },
)

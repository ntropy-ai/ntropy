from setuptools import setup, find_packages

setup(
    name='ntropy',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pydantic==2.8.2',
        'cryptography==42.0.7',
        'tabulate==0.9.0',
        'torch==1.13.0',
        'Pillow==10.4.0'
    ],
    extras_require={
        'aws': ['boto3', 'botocore'],
        'openai': ['openai', 'clip @ git+https://github.com/openai/CLIP.git']
    },
    # Additional metadata
    author='Hugo Le Belzic',
    author_email='hugolebelzic@gmail.com',
    description='Ntropy ai',
    url='https://github.com/HugoLB0/ntropy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

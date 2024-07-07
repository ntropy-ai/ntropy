from setuptools import setup, find_packages

setup(
    name='ntropy',
    version='0.0.1',
    packages=find_packages(include=['ntropy', 'ntropy.*']),
    install_requires=[
        'pydantic==2.8.2',
        'cryptography==42.0.7',
        'tabulate==0.9.0',
        'Pillow==10.4.0'
    ],
    extras_require={
        'providers-aws': [
            'boto3', 
            'botocore'
        ],
        'providers-openai': [
            'openai', 
            'clip @ git+https://github.com/openai/CLIP.git',
            'torch==2.3.1',
            'torchvision==0.18.1'
        ],
        'document-instance-pdf': [
            'pymupdf'
        ],
        'rag-vectorstore-chroma': [
            'chromadb'
        ]
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
    tests_require=['pytest', 'requests'],
    test_suite='tests'
)

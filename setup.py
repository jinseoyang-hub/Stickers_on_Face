from setuptools import setup, find_packages

setup(
    name='my_project',
    version='1.0.0',
    description='Project that uses OpenCV, NumPy, and Pillow',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'Pillow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

from setuptools import setup, find_packages

setup(
    name="jetseg",
    version="1.0.0",
    description="Optimized Human Segmentation Library for Jetson Orin Nano (TensorRT)",
    author="Bigkatoan",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy<2",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
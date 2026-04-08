from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="verdict-weight",
    version="1.0.0",
    author="Andre Byrd",
    author_email="andre.byrd@odingard.com",
    description="VERDICT WEIGHT™ — Context-Adaptive Multi-Source Confidence Synthesis Framework. Patent Pending #64/032,606. 251 tests, 295,000+ scenarios validated.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Odingard/verdict-weight",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    keywords=[
        "confidence calibration",
        "threat intelligence",
        "multi-source fusion",
        "adversarial AI",
        "AI governance",
        "cybersecurity",
        "verdict weight",
    ],
)

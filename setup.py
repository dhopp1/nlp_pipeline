import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlp_pipeline",
    version="0.1.0",
    author="Daniel Hopp",
    author_email="daniel.hopp1@gmail.com",
    description="Pipelines and management structure for NLP analysis of a corpus of texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhopp1/nlp_pipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

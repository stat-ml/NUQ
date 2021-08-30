from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nuq",
    version="0.0.1",
    author="N. Kotelevskii et al.",
    author_email="nikita.kotelevskii@skoltech.ru",
    description="Non-parametric uncertainty estimation method (NUQ)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nkotelevskii/nw_uncertainty",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

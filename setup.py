import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ccnet", # Replace with your own username
    version="1.0.0",
    author="Junbo Jia",
    author_email="junbo_jia@163.com",
    description="A network-based single-cell RNA-seq data analysis library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Just-Jia/ccNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=[setuptools.Extension("ccnet.visualize.tool", ["ccnet/visualize/tool.c"])]
)

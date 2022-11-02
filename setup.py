import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyIAAS",
    version="0.0.1",
    author="JasonLoveDL",
    author_email="2421049459@qq.com",
    description="Code of IAAS Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JasonloveDL/pyIAAS",
    project_urls={
        "Bug Tracker": "https://github.com/JasonloveDL/pyIAAS/issues",
    },
    classifiers=[  # https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    packages=setuptools.find_packages(include=['pyIAAS', 'pyIAAS.*']),
    python_requires=">=3.6",
    py_modules=['main', 'pyIAAS', 'pyIAAS.*'],
    entry_points='''
        [console_scripts]
        pyIAAS=main:cli
    ''',
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchaudio',
        'torchvision ',
        'gym',
        'matplotlib',
        'click'
        'concurrent-log-handler'
      ],

)

import setuptools

setuptools.setup(
    name='genephys',
    version='0.1.5',
    description='Generative model of empirical electrophysiological signals',
    url='https://github.com/vidaurre/genephys',
    author='Diego Vidaurre',
    author_email = "dvidaurre@cfin.au.dk",
    install_requires=['scipy','numpy','sklearn','matplotlib'],
    packages=["genephys"],
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3"]
    )


from setuptools import setup, find_packages
name="spark_ml_utils"
VERSION = '0.0.3' 
DESCRIPTION = '''Some spark ml utilities, for easy checking/modifying spark pipeline, extracting feature importance for spark logistic regression model'''
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


# Setting up
setup(
        name=name, 
        version=VERSION,
        author="Xinyong Tian",
        author_email="<xinyongtian@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
		long_description_content_type="text/markdown",
		url="https://github.com/xinyongtian/py_spark_ml_utils",
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['spark', 'pipeline','feature importance'],
		classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: MIT License",
			"Operating System :: OS Independent",
		]
)
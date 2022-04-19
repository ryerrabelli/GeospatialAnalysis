"""
# Rahul Yerrabelli
#
# Module setup file
# Originally created following this tutorial on versioning levels:
# https://jacobtomlinson.dev/posts/2020/creating-an-open-source-python-project-from-scratch/
# https://jacobtomlinson.dev/posts/2020/versioning-and-formatting-your-python-code/
"""

import setuptools

# Actions in "try:" happen when ryerrabelli module is used to set up any module besides ryerrabelli.
# Actions in "except:" happen when the ryerrabelli module is being setup itself, using the ryerrabelli module.
# Can't use the regular import statements when installing ryerrabelli bc ryerrabelli will not already be installed
# (duh), so can't import it, except by importing it as a relative path To keep the code the same, the renaming of the
# module(s) to "rsy" is helpful. However, for the relative import, we can only have one module named as rsy as we
# need one import statement per module (as opposed to one import statement for the entire package as done when
# ryerrabelli is used external code). Fortunately, each file that imports ryerrabelli only requires functions from
# one other module. The setup.py file only needs to import ryerrabelli functions that are in main.py (doesn't need
# any functions in utils.py) and main.py only needs to import functions from util.py
# ***NOTE: Above message is present in multiple files. Update text in all if comment is changed.
try:
    import ryerrabelli as rsy
except ImportError as e:
    import src.ryerrabelli.packaging as rsy


def setup():

    ####################################################################################################################
    # Below line needs to be changed for every different setup.py file
    module_name = "GeospatialAnalysis"
    ####################################################################################################################


    # Originally, module_data is  outputted as a dict, which makes it easier for printing
    module_data = rsy.get_module_data(module_name=module_name)
    # Operator "**" unpacks dict into named arguments i.e. setup(name=x,version=y, cmdclass=z,...), which is
    # what setuptools.setup(.) requires
    setuptools.setup(**module_data)


if __name__ == '__main__':
    setup()

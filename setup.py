from setuptools import setup, find_packages
from typing import List

# Declaring constants for the project
PROJECT_NAME = "Universal Recommender System"
VERSION = "0.0.1"
AUTHOR = "Lovish Mittal"
DESCRIPTION = "This application will provide recommendations for all your needs."
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements_list() -> List[str]:
    """
    Description: This function is going to return the list of all the packages mentioned in requirements.txt file.

    return: List of packages in requirements.txt file.
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),  # find packages find all packages with init file
    install_requires=get_requirements_list()
)

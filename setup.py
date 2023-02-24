import os.path
import sys
from setuptools import find_packages, setup


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pandaenv"))
extras = {
"gosafeopt": ["gosafeopt @ git+ssh://git@github.com/Data-Science-in-Mechanical-Engineering/Contextual-GoSafe.git"
],
"classified_reg": [
      "botorch==0.3.0",
      "pyyaml",
      "hydra-core==0.11.3",
      "nlopt==2.6.2",
      "classireg @ git+ssh://git@github.com/alonrot/classified_regression.git",
            ],
}
extras["all"] = [item for group in extras.values() for item in group]

setup(name='pandaenv',
      version='0.0.1',
      author='Bhavya Sukhija',
      author_email='sukhijab@ethz.ch',
      packages=[package for package in find_packages() if package.startswith("pandaenv")],
      package_data={'pandaenv': ['envs/assets/Panda_xml/*.stl', 'envs/assets/Panda_xml/*.xml']},
      url='https://github.com/Data-Science-in-Mechanical-Engineering/franka-emika-panda-simulation',
      license='MIT',
      install_requires=[
      'gym<=0.21.0',
      'mujoco_py'],#And any other dependencies required
      extras_require = extras,
      
          classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5'],
)

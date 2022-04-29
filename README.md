# PandaRobot and Simulation Results

## Panda Robot

XML file generated from the URDF and meshes (link: https://github.com/StanfordASL/PandaRobot.jl)

Torque and position controls implemented in the XML file (limits taken from: https://frankaemika.github.io/docs/control_parameters.html#constants)


Can be used with mujoco simulator (recommended version: 2.0.2.8 )

To run a mujoco license is necessary.

Follow the following commands for running the simulation:

1. Enter the mujoco path: ``` ~/mujoco/mujoco200/bin```
```
./simulate /path/to/franka-emika-panda-simulation/Panda_Env/envs/assets/Panda_xml/model_actuate.xml
```

Furthermore, a simple gym environment has also been implemented where the action space consists of the joint torques.

Installation: ```pip install .```

To run gosafeopt experiments: ```pip install .[gosafeopt]```

To import simulation: 
``` 
import Panda_Env 
import gym
env=gym.make("PandaEnvPath-v0")
```

Testing:
``` test_env.py ``` can be run to visualize a simple impedance controller.


## Simulation tasks
We consider 2 tasks:

1. Eight dimensional Task: Reaching a desired positive task
2. Eleven dimensional Task: Path following task.

To run SafeOpt and Contextual GoSafe code is required (EIC additionally requires https://github.com/alonrot/classified_regression ).

A. Running Eight dimensional Task

command: ```python3 Eight_dimension_task/8D_task.py method ```

with method = GoSafeOpt or SafeOpt


B. Running Eleven dimensional Task

command: ```python3 Eleven_dimension_task/11D_task.py method ```

with method = GoSafeOpt or SafeOpt or eic

## Files
1. ```setup.py```: Installation file
2. ```osc_controller.py```: Class which defines functions used for operational space controllers (e.g. Getting jacobian, mass matrix etc.)
3. ```test.py``` and ```test_env_path.py```: File used for testing the 8D task and 11D task environment respectively.
4. ```Eight_dimension_task``` and ```Eleven_dimension_task```: Contains files used to run experiments for 8D and 11D tasks respectively. 

Contributors
-------
URDF files: https://github.com/StanfordASL/PandaRobot.jl

Simulation and Experiments: Bhavya Sukhija

License
-------

The code is licenced under the MIT license and free to use by anyone without any restrictions.

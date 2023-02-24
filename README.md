# PandaRobot and Simulation Results

## Panda Robot

The XML file for the Panda robot simulation is generated from the URDF and meshes (link: https://github.com/StanfordASL/PandaRobot.jl). The torque and position controls are implemented in the XML file (limits taken from: https://frankaemika.github.io/docs/control_parameters.html#constants).
The simulation can be used with the MuJoCo simulator (recommended version: 2.0.2.8, for installation details and licenses see the [MujoCo repository](https://github.com/openai/mujoco-py)).

To run the simulation, navigate to ``` ~/mujoco/mujoco200/bin``` in the terminal and execute 

```
./simulate /path/to/franka-emika-panda-simulation/pandaenv/envs/assets/Panda_xml/model_actuate.xml
```

## Installation and Basic Usage
We implemented a simple Gym environment to interact with the robot. The actions pace consists of the joint torques. The Gym environment can be installed by executing

 ```pip install .```

To be able to also run experiments with GoSafeOpt, execute: ```pip install .[gosafeopt]```

To import simulation: 
``` 
import pandaenv 
import gym
env=gym.make("PandaEnvPath-v0")
```

Testing:
```python3 test_env.py ``` can be run to visualize a simple impedance controller.


## Simulation tasks
We consider 2 tasks:

1. Eight dimensional Task: Reaching a desired positive task
2. Eleven dimensional Task: Path following task.

To run SafeOpt and GoSafeOpt, installation wi code is required (EIC additionally requires https://github.com/alonrot/classified_regression ).

A. Running Eight dimensional Task

command: ```python3 Eight_dimension_task/8D_task.py method ```

replacing method with either GoSafeOpt or SafeOpt


B. Running Eleven dimensional Task

command: ```python3 Eleven_dimension_task/11D_task.py method ```

replacing method with either GoSafeOpt or SafeOpt or EIC (note that for running experiments with EIC (expected improvement with constraints), the [EIC code](https://github.com/alonrot/classified_regression) is required)

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

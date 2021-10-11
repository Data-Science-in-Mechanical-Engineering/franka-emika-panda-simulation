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

Installation: ```pip install -e .```

Testing:
``` test_env.py ``` can be run to visualize a simple impedance controller.


## Simulation tasks
We consider 2 tasks:

1. Eight dimensional: Reaching a desired positive task
2. Path following task.

To run SafeOpt and Contextual GoSafe code is required (EIC additionally requires https://github.com/alonrot/classified_regression ).

A. Running Eight dimensional Task

command: python3 Eight_dimension_task/8D_task.py method 

with method = GoSafeCon or SafeOpt


B. Running Eleven dimensional Task

command: python3 Eleven_dimension_task/11D_task.py method 

with method = GoSafeCon or SafeOpt or eic

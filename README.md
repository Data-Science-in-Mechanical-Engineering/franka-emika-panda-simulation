# PandaRobot

XML file generated from the URDF and meshes (link: https://github.com/StanfordASL/PandaRobot.jl)

Torque and position controls implemented in the XML file (limits taken from: https://frankaemika.github.io/docs/control_parameters.html#constants)


Can be used with mujoco simulator

To run a mujoco license is necessary.

Follow the following commands for running the simulation:

1. Enter the mujoco path: ``` ~/mujoco/mujoco200/bin```
```
./simulate /path/to/PandaRobot/deps/Panda_xml/model_actuate.xml
```

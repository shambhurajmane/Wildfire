# Implementing an A* combinatorial planner and a Probabilistic Roadmap (PRM) sampling-based planner for navigating a firetruck and an arsonist Wumpus

This repo presents the results of implementing an A* combinatorial planner and a Probabilistic Roadmap (PRM)
sampling-based planner for navigating a firetruck and an arsonist Wumpus through a cluttered environment with burning
obstacles. The goal was to compare the performance of these two motion planning approaches in terms of computational
efficiency, path planning optimality, and overall firefighting effectiveness.

##  Method

The environment consisted of a 250m x 250m field with 10% coverage of randomly generated tetromino obstacles. The
firetruck and Wumpus were initialized at opposite corners. The Wumpus moved on a grid using A* to find a path to
adjacent obstacles and set them on fire. The spreading fire dynamics were implemented where burning obstacles ignited all
obstacles within a 30m radius after 10 seconds. The firetruck used a precomputed PRM roadmap to plan kinodynamically
feasible paths to burning obstacles and extinguish them by remaining stationary within 10m for 5 seconds. Five 3600
second simulations were run for each planner.
The performance metrics calculated were: ratio of intact obstacles, ratio of burned obstacles, ratio of extinguished
obstacles, and total computation time for each planner. Goal planning for the Wumpus used a simple strategy of lighting
the closest unburned obstacle, while the firetruck prioritized the most dangerous fires using a heuristic based on the
number of nearby intact obstacles.

We have succesfully built the Bionic Arm with 14 DOF.

![git1](https://github.com/shambhurajmane/Wildfire/blob/main/Hastar.gif)

## Vehicles and Kinematics

We are utilizing an Ackermann steering style vehicle for our robot; specifically modeled after the Mercedes Unimog, a
large off roading capable vehicle.For our kinematics, we assume that we can control the velocity of our drive wheels in v,
and the direction of our steering wheels in ψ. We limit our steering angles to ±60o and attempt to maintain a speed of
10 m
s , or 22mph. We assume an L = 2.8 meters.
![git2](https://github.com/shambhurajmane/Wildfire/blob/main/firetruck_kinematics.png)

## Conclusion

The grid-based A* planner resulted in faster computation times but less optimal firefighting ability compared to the PRM
sampling-based method. The ability to pre-compute the roadmap was advantageous, but the topology still constrained
the firetruck’s movements. The simplistic Wumpus goal planning worked well for spreading fires quickly. More advanced
prioritization of fire threats could improve the firetruck’s performance. Overall, this simulation demonstrated trade-
offs between combinatorial and sampling-based planners in path optimality, computational complexity, and dynamic
replanning abilities.
Check report for the details. 
![report](https://github.com/shambhurajmane/Wildfire/blob/main/Wildfire.pdf)

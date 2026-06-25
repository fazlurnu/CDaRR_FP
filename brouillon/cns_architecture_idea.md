from bluesky, we get objects called traffic that has a lot of var: lat, lon, gs, trk, etc. The idea for the CNS system is that it is separated into two: sensor (this is the own measurement of those variables, with some sensor noises using assuming a certain distribution). The other one is adsl, this is the surveillance message, basically things that is in the sensor is sent to other vehicles through surveillance message. On top of measurement noise, this one has a reception probability. this reception probability is essentialy the probability of a message getting updated or not.

The noise model can accomodate diff distribution, for instance gaussian, t-student, biased gaussian etc. And it should also allow a "correlated" position-velocity sampling. For now let's consider only the gaussian and biased-gaussian, but spec is clear.

then, we will use these info by adding them into the traffic object, so something like

traffic.sensor
traffic.adsl

among those, there will be traffic.sensor.pos_acc and traffic.sensor.vel_acc which describe the 95% confidence interval "observed" or used for the noise.

the traffic.sensor and traffic.adsl will then be used in the cd, cr, and crr algorithms like: ownship.sensor (because this is what the ownship observe), and intruder.adsl (this what we see from intruder).

But wait, what if the reception probability is diff between aircraft. For instnace, A looking at B can have reception prob of 0.95. But C looking at B can have reception prob of 0.92. Then the adsl has more dimension no? How to address this?
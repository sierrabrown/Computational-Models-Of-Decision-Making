This folder contains a step-by-step analysis of three neural models implemented in matlab.

1. AI Maze

	This model simulates learning which occurs in the ventral striatum and amygdala, by assigning a positive or negative reward value to each possible move the 'rat' has available. 

	* Read this for an introduction to reward based learning in the ventral striatum and amygdala, which while necessarily simplified provides a foundation for more complicated concepts.

2. Feedback Control : Implemented through a PID Controller

	Thi model explores the effects of feedback on action control. Unlike in robotics, the length of nerves causes transmission of signals to be slow and therefore necessitates a feedback control system, the strength of which greatly determines ones ability to learn precise movements.
	
	While we have established that negative feedback control will not produce perfect movements due to lags, gains, and a level of non-elimnable error at the end of the process, we will explore how exactly it is that the feedback lag values affect performance.

	* Read this for an explanation of how a PID controller works, along with a detailed explanation of the key factors which must be implemented when building one programatically. Useful for all engineering disciplines.

3. Adaptive force generation for preicision-grip lifting by a spectral timing model of the cerebellum

	The “Adaptive force generation for precision-grip lifting by a spectral timing model of the cerebellum” by Ulloa, A., Bullock, D., and Rhodes, B, illustrates how the process of grasping and lifting an object is controlled. The agent must apply a weight and texture dependent grip force that applies the minimal amount of pressure to hold the object, while also applying the appropriate load force to either lift, or lower the object. The model explains this phenomena through the process of learned slip-compensation mediated by the cerebellum.

	* Read this to better understand the paper “Adaptive force generation for precision-grip lifting by a spectral timing model of the cerebellum” by Ulloa, A., Bullock, D., and Rhodes, B. While the system is quite extensive, it is broken down into three manageable subsytems, and after understanding this analysis one will have a detailed understanding of how grip control works, a concept which can be appleid to many other learning processes.
# Cab-pickup-recommendation
Reinforcement Learning

This experiment demonstrates Deep DQN architecutre to solve RL based problems.
In this experiment, the goal is to suggest optimal rides to cab drivers and maximize their profits.
This optimisation problem is formulated as a Markov Decision Process and the state equations are solved by Q learning
Learn only Q(s,a) by always taking best possible action --> Defined as the action that yields the max reward when taken in a state.
For efficient learning, Epsilon Greedy Policy is implemented and that is controlled by parameter

The Environment generates episodes for the agent to learn.
For simplicity puropses, state is defined as f(Day of week, TIme of Day)
Cab demand is modelled as poisson distribution with different means for days in week
Other complex distributions can also be introduced using pymc3 library

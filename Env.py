import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise action space, state space and a random state to start simulating"""
        
        ## Actions which the agent can take (Ensure pickup and drop are not in same city)
        self.pickup = np.random.choice(np.arange(m))
        self.drop = np.random.choice(np.arange(m))
        self.action_space= [(p,q) for p in range(m) for q in range(m) if p!=q]
        self.action_space.insert(0, (0,0)) ## (0,0) implies ride refusal
        
        ## State can be defined in terms of day and hour
        self.state_space =  self.state_space = [(loc,time,day) for loc in range(m) for time in range(t) for day in range(d)]
        
        ## Initializing a random state
        self.state_init = self.state_space[np.random.choice(len(self.state_space))] 

        ## Start the first round
        self.reset()

    def state_encod_arch(self, state):
        """
        Converts a given state into a vector format
        
        """
        
        ## Take the initialized state and convert to vector
        curr_loc, curr_time, curr_day= state

        ## Initialize arrays
        loc_arr = np.zeros(m, dtype=int)
        time_arr= np.zeros(t, dtype=int)
        day_arr= np.zeros(d, dtype= int)

        ## One hot Encoding respective arrays
        loc_arr[curr_loc] = 1
        time_arr[curr_time] = 1
        day_arr[curr_day] = 1

        ## Horizontal stacking of the encoded vectors
        state_encod= np.hstack((loc_arr, time_arr, day_arr))
        state_encod= state_encod.tolist()

        return state_encod

    
    def requests(self, state):
        """
        Determining the number of requests basis the location. 
        Also select only possible action spaces
        """
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
            
        if location == 1:
            requests = np.random.poisson(12)
            
        if location == 2:
            requests = np.random.poisson(4)
            
        if location == 3:
            requests = np.random.poisson(7)
            
        if location == 4:
            requests = np.random.poisson(8)
            
        ## Maximum possible requests
        if requests >15:
            requests =15
        
        ## Pickup and drop cannot be in the same city
        ## We have used (0,0) in the first index of the list to indicate refusals
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0]  
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   
    

    def update_time_day(self, curr_time, curr_day, ride_duration):
        """
        Takes in the current time, current day and duration taken for driver's journey and returns
        updated time and updated day post that journey.
        Since episodes length will be 30  days, the time between trips needs to be tracked for reward calculation.
        Hence we create this tracker method
        """
        ride_duration = int(ride_duration)

        if (curr_time + ride_duration) < 24:
            updated_time = curr_time + ride_duration  
            updated_day= curr_day # Same Day
        else:
            # duration spreads over to next day
            updated_time = (curr_time + ride_duration) % 24 
            num_days = (curr_time + ride_duration) // 24
            updated_day = (curr_day + num_days ) % 7

        return updated_time, updated_day
    
    
    def get_next_state_and_time_func(self, state, action, Time_matrix):
        """
        Takes state, action and Time_matrix as input and returns next state and different times during the journey.
        """
        
        next_state = []
        
        # Initialize times
        total_time   = 0
        transit_time = 0         # To go from current location to pickup location
        wait_time = 0    # in case driver chooses to refuse all requests. for action: (0,0) 
        ride_time    = 0         # From Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc, curr_time, curr_day = state
        pickup_loc, drop_loc= action
        
        if ((pickup_loc== 0) and (drop_loc == 0)): ## Driver refuses request
            wait_time = 1    
            next_loc = curr_loc
            
        elif (curr_loc == pickup_loc): ## Driver is already at pickup spot
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            next_loc = drop_loc
            
        else:
            # Driver is not at the pickup spot. He/she needs to commute to the pickup spot from the curr_loc
            # Time take to reach pickup spot (from current location to pickup spot)
            transit_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)
            
            # The cab driver is now at the pickup spot
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)
        
        # Finding next_state using the next_loc and the next time states.
        next_state = [next_loc, next_time, next_day]
        
        return next_state, wait_time, transit_time, ride_time
    
    
    ## State transition    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state, action and Time_matrix as input and returns next state"""
        next_state= self.get_next_state_and_time_func(state, action, Time_matrix)[0]   ## get_next_state_and_time_func() defined above       
        return next_state

    
    def reward_func(self, state, action, Time_matrix):
        """
        Takes in state, action and Time_matrix and returns the reward
        """
        ## get_next_state_and_time_func() defined above 
        wait_time, transit_time, ride_time = self.get_next_state_and_time_func(state, action, Time_matrix)[1:]
        
        pickup_loc, drop_loc = action
        
        # transit and wait time yield no revenue and consumes battery
        idle_time = wait_time + transit_time
        customer_ride_time = ride_time
        
        earnings = (R * customer_ride_time)
        costs = (C * (customer_ride_time + idle_time))
        reward = earnings - costs
        
        if ((pickup_loc== 0) and (drop_loc == 0)): ## Driver refuses request
            reward = -C
            
        return reward
    
    def step(self, state, action, Time_matrix):
        """
        Take a trip as a cab driver. Takes state, action and Time_matrix as input and returns next_state, reward and total time spent
        """
        # Get the next state and the various time durations
        next_state, wait_time, transit_time, ride_time = self.get_next_state_and_time_func(state, action, Time_matrix)

        # Calculate the reward and total_time of the step
        reward = self.reward_func(state, action, Time_matrix)
        total_time = wait_time + transit_time + ride_time
        
        return next_state, reward, total_time

    
    def reset(self):
        return self.action_space, self.state_space, self.state_init
    

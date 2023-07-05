import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_checker import check_env

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = 38 + random.randint(-3,3)
        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self, mode=0):
        # Implement viz
        if self.state <= 39 and self.state >= 37:
            print(f"Shower is {self.state}°C at the moment, nice temp! +1")
        else:
            print(f"Shower is {self.state}°C at the moment, too hot or cold! -1")
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        # Reset shower time
        self.shower_length = 60 
        return self.state
    

def main():

    if True:
        env = ShowerEnv()
        model = PPO.load('PPO_shower')
        print(evaluate_policy(model, env, n_eval_episodes=10, render=False))
        episodes = 5
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            score = 0 
            
            while not done:
                env.render()
                action, _states = model.predict(state)
                state, reward, done, info = env.step(action)
                score+=reward
            print('Episode:{} Score:{}'.format(episode, score))
        env.close()
        return

    env=ShowerEnv()

    env.reset()

    #check_env(env, warn=True)



    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()


    log_path = os.path.join('.', 'Training', 'Logs')
    print(log_path)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)

    model.learn(total_timesteps=1000000)

    model.save('PPO_shower')
    print("Model Saved")

    print(evaluate_policy(model, env, n_eval_episodes=10, render=False))





if __name__ ==  "__main__":
    main()
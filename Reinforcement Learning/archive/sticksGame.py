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

from sys import argv

class StickGame(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Discrete(100)
        # Set start temp
        self.state = 99
        self.model = None
        
    def step(self, action):

        reward = 0
        done = False
        
        # Calculate reward
        self.state -= action + 1


        if self.state <= 0:
            done = True
            reward = -1
        else:
            # opponent turn
            if self.model != None:                
                op_action, _states = self.model.predict(self.state)
                self.state -= op_action + 1
            else:
                self.state -= random.randint(1,3)
            if self.state <= 0:
                reward = 1
                done = True
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self, mode=0):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = 99
        try:
            self.model = PPO.load('PPO_sticks_random_trainedd')
        except:
            self.model = None
        return self.state
    

def main():

    a = False
    if len(argv) > 1 and argv[1] == "eval":
        a = True
    if a == True:
        env = StickGame()
        model = PPO.load('PPO_sticks_random_trained')
        print(evaluate_policy(model, env, n_eval_episodes=100, render=False))
        episodes = 1000
        lost = 0
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            score = 0 
            
            while not done:
                env.render()
                action, _states = model.predict(state)
                state, reward, done, info = env.step(action)
                score+=reward
            if score == -1:
                lost += 1
            #print('Episode:{} Score:{}'.format(episode, score))
        print("Lost: ", lost)
        env.close()
        return

    env=StickGame()

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
    for i in range(10):
        env.reset()
        try:
            model = PPO.load('PPO_sticks_random_trained', env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)
        except:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)

        model.learn(total_timesteps=100000)

        model.save('PPO_sticks_random_trained')
        print("Model Saved")

    print(evaluate_policy(model, env, n_eval_episodes=100, render=False))





if __name__ ==  "__main__":
    main()
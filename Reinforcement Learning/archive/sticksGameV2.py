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

    STICKS = 60
    MODEL = 'PPO_sticksV2_0'

    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = MultiDiscrete([3,StickGame.STICKS])
        # Temperature array
        self.observation_space = Box(low=0, high=1, dtype=np.int8, shape=(StickGame.STICKS,))
        # Set start temp
        self.reset()
        self.count = StickGame.STICKS
        
    def step(self, action):

        reward = 0
        done = False
        info = {"win": 0, "lose_by_illegal": 0, "sticks_left": self.count}

        stick_nb = action[0] + 1
        
        #illegal move
        if action[1] + stick_nb >= StickGame.STICKS or len([x for x in self.state[action[1]:action[1]+stick_nb] if x == 0]) > 0:
            reward = -100000
            return self.state, reward, done, info
        else:
            #player turn
            self.state[action[1]:action[1]+stick_nb] = [0] * stick_nb
            self.count -= stick_nb
            reward = 5


        if self.count <= 0:
            done = True
            reward = -100
        else:
            # opponent turn random
            try:
                model = PPO.load(StickGame.MODEL)
                first_try = True

                # #illegal move
                # while first_try or op_action[1] + op_nb >= StickGame.STICKS or len([x for x in self.state[op_action[1]:op_action[1]+op_nb] if x == 0]) > 0:
                #     if op_illegal == 0:
                
                #         return self.state, reward, done, info
                #     state = self.state
                #     op_action, _states = model.predict(state)
                #     op_nb = op_action[0] + 1
                #     first_try = False
                # #opponent turn
                # self.state[op_action[1]:op_action[1]+op_nb] = [0] * op_nb
                # self.count -= op_nb
                pass

            except:
                possible_pos = []
                for i in range(0, StickGame.STICKS):
                    if self.state[i] == 1:
                        possible_pos.append(i)
                pos = random.choice(possible_pos)
                nb = random.randint(1,len([x for x in self.state[pos:pos+3] if x == 1]))
                self.state[pos:pos+nb] = [0] * nb
                self.count -= nb
                
            if self.count <= 0:
                info["win"] = 1
                reward = 100
                done = True
        
        # Return step information
        info["sticks_left"] = self.count
        return self.state, reward, done, info

    def render(self, mode=0):
        #print(self.state)
        pass
    
    def reset(self, model_i = 0):
        # Reset shower temperature
        self.state = np.array([1 for x in range(StickGame.STICKS)]).astype(np.int8)
        self.count = StickGame.STICKS
        StickGame.MODEL = "PPO_sticksV2_" + str(model_i)
        return self.state
    

def main():
    a = False
    if len(argv) > 1 and argv[1] == "eval":
        a = True
    if a == True:
        print("Eval mode")
        env = StickGame()
        model = PPO.load('PPO_sticksV2_1')
        print(evaluate_policy(model, env, n_eval_episodes=10, render=False))
        episodes = 100
        lost = 0
        illegal = 0
        for episode in range(1, episodes+1):
            state = env.reset(0)
            done = False
            score = 0 
            
            while not done:
                env.render()
                action, _states = model.predict(state)
                state, reward, done, info = env.step(action)
                score+=reward
            if info["win"] == 0:
                lost += 1
            if info["lose_by_illegal"] == 1:
                illegal += 1
            print(f"Episode:{episode} Score:{score}, sticks left: {info['sticks_left']}")
        print(f"Lost: {lost}, illegal: {illegal}")
        env.close()
        return

    env=StickGame()

    env.reset()

    #check_env(env, warn=True)


    log_path = os.path.join('.', 'Training', 'Logs')
    print(log_path)

    #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)
    for i in range(1, 3):
        model_name = 'PPO_sticksV2_' + str(i)
        env.reset(i)
        model = PPO.load(model_name, env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)
        model.learn(total_timesteps=100000)

        model.save('PPO_sticksV2_' + str(i+1))
        print("Model Saved")

        print(evaluate_policy(model, env, n_eval_episodes=10, render=True))





if __name__ ==  "__main__":
    main()
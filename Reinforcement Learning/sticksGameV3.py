import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_checker import check_env

from sys import argv

class StickGame(Env):

    STICKS = 12
    WIN_REWARD = 10

    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = MultiDiscrete([3,StickGame.STICKS])
        # Temperature array
        self.observation_space = Box(low=0, high=1, dtype=np.int8, shape=(StickGame.STICKS,))
        # Set start temp
        self.reset()
        self.count = StickGame.STICKS
        self.play = False
        self.begin = True
        self.op_model = "0"
        
    def step(self, action):

        reward = 0
        done = False
        info = {"win": 0, "sticks_left": self.count}

        stick_nb = action[0] + 1
        
        #illegal move
        if self.count < StickGame.STICKS or self.begin:
            if action[1] + stick_nb >= StickGame.STICKS or len([x for x in self.state[action[1]:action[1]+stick_nb] if x == 0]) > 0:
                reward = -(StickGame.WIN_REWARD + 90)
                done = True
                return self.state, reward, done, info
            else:
                #player turn
                self.state[action[1]:action[1]+stick_nb] = [0] * stick_nb
                self.count -= stick_nb


        if self.count <= 0:
            done = True
            reward = -StickGame.WIN_REWARD
        else:
            # opponent turn random
            if self.play:
                print(self.state)
                nb = 0
                pos = 0
                while nb < 1 or nb > 3 or pos < 0 or pos + nb > StickGame.STICKS or len([x for x in self.state[int(pos):int(pos)+int(nb)] if x == 0]) > 0:
                    pos = int(input("Which position do you want to remove sticks from? ")) - 1
                    nb = int(input("How many sticks do you want to remove? "))
                self.state[int(pos):int(pos)+int(nb)] = [0] * int(nb)
                self.count -= int(nb)
            else:
                try:
                    model = PPO.load(self.op_model)

                    #illegal move
                    state = self.state
                    op_action, _states = model.predict(state)
                    op_nb = op_action[0] + 1
                    if op_action[1] + op_nb >= StickGame.STICKS or len([x for x in self.state[op_action[1]:op_action[1]+op_nb] if x == 0]) > 0:
                        possible_pos = []
                        for i in range(0, StickGame.STICKS):
                            if self.state[i] == 1:
                                possible_pos.append(i)
                        pos = random.choice(possible_pos)
                        possible_nb = 0
                        for i in self.state[pos:pos+3]:
                            if i == 0:
                                break
                            else:
                                possible_nb += 1
                        nb = random.choice(range(possible_nb)) + 1
                        self.state[pos:pos+nb] = [0] * nb
                        self.count -= nb
                    else:
                    #opponent turn
                        self.state[op_action[1]:op_action[1]+op_nb] = [0] * op_nb
                        self.count -= op_nb

                except:
                    possible_pos = []
                    for i in range(0, StickGame.STICKS):
                        if self.state[i] == 1:
                            possible_pos.append(i)
                    pos = random.choice(possible_pos)
                    possible_nb = 0
                    for i in self.state[pos:pos+3]:
                        if i == 0:
                            break
                        else:
                            possible_nb += 1
                    nb = random.choice(range(possible_nb)) + 1
                    self.state[pos:pos+nb] = [0] * nb
                    self.count -= nb
                
            if self.count <= 0:
                info["win"] = 1
                reward = StickGame.WIN_REWARD
                done = True
        
        # Return step information
        info["sticks_left"] = self.count
        return self.state, reward, done, info

    def render(self, mode=0):
        #print(self.state)
        pass
    
    def reset(self, model = "0", play=False, begin="random"):
        # Reset shower temperature
        if begin == "random":
            self.begin = random.choice([True, False])
        else:
            self.begin = begin
        self.play = play
        self.state = np.array([1 for x in range(StickGame.STICKS)]).astype(np.int8)
        self.count = StickGame.STICKS
        self.op_model = model
        return self.state
    

def main():
    a = False
    if len(argv) > 1:
        if argv[1] == "eval":
            print("Eval mode")
            env = StickGame()
            model = PPO.load('PPO_sticksV3_0')
            print(evaluate_policy(model, env, n_eval_episodes=10, render=False))
            episodes = 100
            lost = 0
            for episode in range(1, episodes+1):
                state = env.reset("PPO_sticksV3_")
                done = False
                score = 0 
                
                while not done:
                    env.render()
                    action, _states = model.predict(state)
                    state, reward, done, info = env.step(action)
                    score+=reward
                if info["win"] == 0:
                    lost += 1
                print(f"Episode:{episode} Score:{score}, Sticks left:{info['sticks_left']}")
            print(f"Lost {lost} out of {episodes} games")
            env.close()
            return
        elif argv[1] == "play":
            print("Play mode")
            env = StickGame()
            model = PPO.load('PPO_sticksV3_4')
            state = env.reset(play=True)
            done = False
            score = 0 
            while not done:
                env.render()
                action, _states = model.predict(state)
                state, reward, done, info = env.step(action)
                score+=reward
            print(f"Score:{score}")
            env.close()
            return
        elif argv[1] == "train":

            env=StickGame()

            env.reset()

            #check_env(env, warn=True)


            true_log_path = os.path.join('.', 'Training', 'Logs')
            print(true_log_path)

            # for i in range(1, 2):
            #     j = i + 1
            #     log_path = true_log_path
            
            k = 0
            while True:
                k += 1
                i = 0
                j = 0
                if k % 10 == 0:
                    log_path = true_log_path
                else:
                    log_path = None

                model_name = 'PPO_sticksV3_' + str(i)
                env.reset(model="PPO_sticksV3_" + str(i))
                try:
                    model = PPO.load(model_name, env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)
                except:
                    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)
                model.learn(total_timesteps=1000000)

                model.save('PPO_sticksV3_' + str(j))
                print("Model Saved")

                print(evaluate_policy(model, env, n_eval_episodes=10, render=True))



# entrainement à x/60 (ou toujours 60 mais possibilité de démarer à x/60)
# random sur qui commence la game (attention à donner l'état du jeu avant de jouer le coup)

if __name__ ==  "__main__":
    main()
from flask import Flask, render_template, request, flash
import sys
from stable_baselines3 import PPO


app = Flask(__name__)
app.secret_key = "sfekjnJFJQFHFEQJSFhbsqenfg54d54daq4"

# def illegal_move(s_sticks):
#     if len(s_sticks) == 0:
#         return("You must select at least one stick")
#     if len(s_sticks) > 3:
#         return("You can only select 3 sticks")
#     if len(s_sticks) == 3:
#         if s_sticks[0] + 1 != s_sticks[1] or s_sticks[1] + 1 != s_sticks[2]:
#             return("You can only select consecutive sticks")
#     if len(s_sticks) == 2:
#         if s_sticks[0] + 1 != s_sticks[1]:
#             return("You can only select consecutive sticks")
#     return False

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        state = request.form.getlist("state[]")
        print(state)
        if state == None or len(state) != 12:
            return (render_template('index.html'))
        state = [int(i) for i in state]
        model = PPO.load('./models/PPO_sticksV3')
        action, _states = model.predict(state)

        nb = action[0] + 1
        pos = action[1]
        print(action)
        return ({"nb":str(nb), "pos":str(pos), "error":""})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
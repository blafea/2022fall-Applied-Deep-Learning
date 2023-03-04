import matplotlib.pyplot as plt
import json

with open("trainer_state.json", "r") as f:
    file = json.load(f)["log_history"]

step = []
rouge1 = []
rouge2 = []
rougel = []
for i in file:
    if len(i) == 10:
        step.append(i["step"])
        rouge1.append(i["eval_rouge-1"])
        rouge2.append(i["eval_rouge-2"])
        rougel.append(i["eval_rouge-l"])

plt.figure()
plt.plot(step, rouge1, label="rouge-1_f")
plt.plot(step, rouge2, label="rouge-2_f")
plt.plot(step, rougel, label="rouge-l_f")
plt.xlabel("steps")
plt.ylabel("F1 scores")
plt.title("Learning Curves", fontsize=18)
plt.legend()
plt.show()
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from gridworld.environment_momentum import Environment
from agent.base_agent import BaseAgent
from algorithm.qpsp import QPSPAlgorithm

from logfunc.callback import LoggerCallback

ENV_NUM = 3
AGENT_NUM = 5
ALPHA = 0.06
GAMMA = 0.95
D_RATE = 0.20
TEMPERATURE = 0.52
#PAST_N = 2
PAST_N = None
MAX_STEPS = 1000000
BLOCK = 1000
#SEEDS = [0]
SEEDS = [0,1,2,3,4,5,6,7,8,9]

OUTF = './result'
now = datetime.now()
NOW = now.strftime("%Y%m%d-%H%M")
OUTF = os.path.join(OUTF,NOW)

try:
    os.makedirs(OUTF)
except FileExistsError:
    pass

log_file_path = os.path.join(OUTF,f'result_env={ENV_NUM}_agent={AGENT_NUM}_pastN={PAST_N}.txt')

plt.figure()

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    env = Environment(AGENT_NUM, ENV_NUM, PAST_N)
    agents = [BaseAgent() for _ in range(AGENT_NUM)]
    algo = QPSPAlgorithm(alpha=ALPHA, gamma=GAMMA, d_rate=D_RATE)

    callback = LoggerCallback(log_file_path, AGENT_NUM, PAST_N)

    history = algo.run(env, agents, TEMPERATURE, MAX_STEPS, callback, block=BLOCK)

    callback.on_seed_end(seed, history)

    steps = [(i+1) * BLOCK for i in range(len(history))]

    label = f"seed={seed}"
    plt.plot(steps, history, label=label)

    with open(log_file_path, 'a') as f:
        f.write(f'{history}\nseed={seed} done\n')

    plt.savefig(f'{OUTF}/result_env_{ENV_NUM}_agents={AGENT_NUM}_n={PAST_N}.png')

plt.xlabel("step")
plt.ylabel("deliveries per 1000 steps")
plt.title(f"agents={AGENT_NUM}")
plt.grid(True,linestyle="--", alpha=0.6)
plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0), borderaxespad=0)
plt.tight_layout()

plt.savefig(f'{OUTF}/result_env_{ENV_NUM}_agents={AGENT_NUM}_n={PAST_N}.png')
print("シミュレーション全終了")
plt.show()

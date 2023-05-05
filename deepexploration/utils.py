## Import all dependencies for file
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from easyrl.configs import cfg
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.value_net import ValueNet
from easyrl.models.mlp import MLP
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import save_traj
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import freeze_model
from easyrl.utils.torch_util import load_state_dict
from easyrl.utils.torch_util import move_to
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np


@dataclass
class BasicAgent:
    actor: nn.Module

    def __post_init__(self):
        move_to([self.actor],
                device=cfg.alg.device)

    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        t_ob = torch_float(ob, device=cfg.alg.device)
        # the policy returns a multi-variate gaussian distribution
        act_dist, _ = self.actor(t_ob)
        # sample from the distribution
        action = action_from_dist(act_dist,
                                  sample=sample)
        # get the log-probability of the sampled actions
        log_prob = action_log_prob(action, act_dist)
        # get the entropy of the action distribution
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy),
        )
        return torch_to_np(action), action_info


def create_actor(env):
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor_body = MLP(
        input_size=ob_dim,
        hidden_sizes=[64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh
    )
    actor = DiagGaussianPolicy(actor_body,
                               in_features=64,
                               action_dim=action_dim)
    return actor


def create_critic(env):
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    critic_body = MLP(
        input_size=ob_dim,
        hidden_sizes=[64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh
    )

    critic = ValueNet(critic_body, in_features=64)

    return critic


# def initialize_critic(critic, trajs, num_epochs=10, batch_size=64, lr=1e-3):
#     optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
#     for epoch in range(num_epochs):
#         for traj in trajs:
#             ob = traj.obs
#             ret = traj.raw_rewards
#             for i in range(ob.shape[1]):
#                 optimizer.zero_grad()
#                 ob_i = ob[:, i, :]
#                 ret_i = ret[:, i]
#                 ret_i = torch_float(ret_i)
#                 ret_i = ret_i.unsqueeze(1)
#                 ret_i = move_to([ret_i], device=cfg.alg.device)[0]
#                 ret_pred = critic(ob_i)
#                 loss = loss_fn(ret_pred, ret_i)
#                 loss.backward()
#                 optimizer.step()
#     return critic

def load_expert_agent(env, device, expert_model_path='pusher_expert_model.pt'):
    expert_actor = create_actor(env=env)
    expert_agent = BasicAgent(actor=expert_actor)
    print(f'Loading expert model from: {expert_model_path}.')
    ckpt_data = torch.load(expert_model_path, map_location=torch.device(f'{device}'))
    load_state_dict(expert_agent.actor,
                    ckpt_data['actor_state_dict'])
    freeze_model(expert_agent.actor)
    return expert_agent


def generate_demonstration_data(expert_agent, env, num_trials):
    return run_inference(expert_agent, env, num_trials, return_on_done=True)


def run_inference(agent, env, num_trials, return_on_done=False, sample=True, disable_tqdm=False, render=False):
    runner = EpisodicRunner(agent=agent, env=env)
    trajs = []
    for _ in tqdm(range(num_trials), desc='Run', disable=disable_tqdm):
        env.reset()
        traj = runner(time_steps=cfg.alg.episode_steps,
                      sample=sample,
                      return_on_done=return_on_done,
                      evaluation=True,
                      render_image=render)
        trajs.append(traj)
    return trajs


def eval_agent(agent, env, num_trials, disable_tqdm=False, render=False):
    trajs = run_inference(agent, env, num_trials, return_on_done=True,
                          disable_tqdm=disable_tqdm, render=render)
    successes = []
    rets = []
    for traj in trajs:
        tsps = traj.steps_til_done.copy().tolist()
        rewards = traj.raw_rewards
        infos = traj.infos
        for ej in range(rewards.shape[1]):
            ret = np.sum(rewards[:tsps[ej], ej])
            rets.append(ret)
            successes.append(infos[tsps[ej] - 1][ej]['success'])
        if render:
            save_traj(traj, 'tmp')
    ret_mean = np.mean(rets)
    ret_std = np.std(rets)
    success_rate = np.mean(successes)
    return success_rate, ret_mean, ret_std, rets, successes


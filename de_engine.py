from itertools import count
from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import torch
import torch.optim as optim
from easyrl.configs import cfg
import tqdm
from easyrl.utils.common import save_traj
from utils import create_actor
from utils import load_expert_agent
from utils import generate_demonstration_data
from utils import run_inference
from utils import eval_agent


class TrajDataset(Dataset):
    def __init__(self, trajs, sample_percent=.9):
        states = []
        actions = []
        for traj in trajs:
            states.append(traj.obs)
            actions.append(traj.actions)

        # sample a fraction of the dataset
        # rand_indices = np.random.choice(len(states), size=np.floor(states*sample_percent))
        # self.states = np.concatenate([states[i] for i in rand_indices], axis=0)
        # self.actions = np.concatenate([actions[i] for i in rand_indices], axis=0
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        sample = dict()
        sample['state'] = self.states[idx]
        sample['action'] = self.actions[idx]
        return sample

    def add_traj(self, traj=None, states=None, actions=None):
        if traj is not None:
            self.states = np.concatenate((self.states, traj.obs), axis=0)
            self.actions = np.concatenate((self.actions, traj.actions), axis=0)
        else:
            self.states = np.concatenate((self.states, states), axis=0)
            self.actions = np.concatenate((self.actions, actions), axis=0)

    def get_sample(self, sample_ratio=.9):
        dataset_size = len(self.states)
        rand_indices = np.random.choice(dataset_size, size=np.floor(dataset_size * sample_ratio))
        return




def train_de_agent(agent, trajs, max_epochs=5000, batch_size=256, lr=0.0005, disable_tqdm=True):
    dataset = TrajDataset(trajs)

    #TODO: override DataLoader default behavior to sample randomly with replacement from the dataset
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)

    optimizer = optim.Adam(agent.actor.parameters(),
                           lr=lr)
    pbar = tqdm(range(max_epochs), desc='Epoch', disable=disable_tqdm)
    logs = dict(loss=[], epoch=[])
    for iter in pbar:
        avg_loss = []
        for batch_idx, sample in enumerate(dataloader):
            states = sample['state'].float().to(cfg.alg.device)
            expert_actions = sample['action'].float().to(cfg.alg.device)
            optimizer.zero_grad()
            act_dist, _ = agent.actor(states)
            #### TODO: compute the loss in a variable named as 'loss'
            #### using the act_dist and expert_actions
            loss = -torch.sum(act_dist.log_prob(expert_actions))
            ####
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            avg_loss.append(loss.item())
        logs['loss'].append(np.mean(avg_loss))
        logs['epoch'].append(iter)
    return agent, logs, len(dataset)

def eval_agent(agent, env, num_trials, disable_tqdm=False, render=False):
    trajs = run_inference(agent, env, num_trials, return_on_done=True,
                          disable_tqdm=disable_tqdm, render=render)
    tsps = []
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

@dataclass
class DeepExplorationEngine:
    agent: Any
    runner: Any
    env: Any
    trajs: Any

    def __post_init__(self):
        self.dataset = TrajDataset(self.trajs)

    def train(self):
        success_rates = []
        dataset_sizes = []
        self.cur_step = 0
        for iter_t in count():
            print(f'iter: {iter_t}')
            if iter_t % cfg.alg.eval_interval == 0:
                success_rate, ret_mean, ret_std, rets, successes = eval_agent(self.agent,
                                                                              self.env,
                                                                              200,
                                                                              disable_tqdm=True)
                success_rates.append(success_rate)
                dataset_sizes.append(len(self.dataset))
            # rollout the current policy and get a trajectory
            # TODO: rollout multiple trajectories
            traj = self.runner(sample=True, get_last_val=False, time_steps=cfg.alg.episode_steps)
            # optimize the policy
            self.train_once(traj)
            if self.cur_step > cfg.alg.max_steps:
                break
        return dataset_sizes, success_rates

    def train_once(self, trajs, sample_ratio):
        self.cur_step += traj.total_steps # TODO: figure out how to calculate steps

        # action_infos = traj.action_infos
        # exp_act = torch.stack([ainfo['exp_act'] for ainfo in action_infos])

        # Actually we're going to instantiate
        # TODO: clear out dataset?
        for traj in trajs:
            action_infos = traj.action_infos
            exp_act = torch.stack([ainfo['exp_act'] for ainfo in action_infos])
            self.dataset.add_traj(states=traj.obs,
                                    actions=exp_act.cpu())
        dataset_size = len(self.dataset.states)
        # do this for each of our k value functions
        rand_indices = np.random.choice(dataset_size, size=np.floor(dataset_size * sample_ratio))
        rollout_dataloader = DataLoader(Subset(self.dataset, rand_indices),
                                        batch_size=cfg.alg.batch_size,
                                        shuffle=True,
                                        )
        # TODO - update value function with rollout_dataloader stuff
        optim_infos = []
        for oe in range(cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize(batch_data)
                optim_infos.append(optim_info)
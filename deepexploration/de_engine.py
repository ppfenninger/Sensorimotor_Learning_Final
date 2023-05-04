import time
from itertools import count
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
from easyrl.utils.gae import cal_gae, cal_gae_torch
import numpy as np
import torch
import torch.optim as optim
from easyrl.configs import cfg
import tqdm
from easyrl.utils.common import save_traj
from deepexploration.utils import run_inference
from deepexploration.utils import eval_agent
from typing import Any


class TrajDataset(Dataset):
    def __init__(self, trajs):
        states = []
        actions = []
        values = []
        advantages = []
        returns = []
        log_probs = []

        for traj in trajs:
            
            obs, a, adv, ret, log_prob, val = self.traj_preprocess(traj)
            states.append(obs)
            actions.append(a)
            values.append(val)
            log_probs.append(log_prob)
            returns.append(ret)
            advantages.append(adv)

        self.data = {
            "ob": np.concatenate(states, axis=0),
            "action": np.concatenate(actions, axis=0),
            "val": np.concatenate(values, axis=0), 
            "adv": np.concatenate(advantages, axis=0),
            "ret": np.concatenate(returns, axis=0), 
            "log_prob": np.concatenate(log_probs, axis=0)
        }


    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        sample = dict()
        for key in self.data: 
            sample[key] = self.data[key][idx]
        return sample

    def add_traj(self, traj=None, states=None, actions=None):
        if traj is not None:
            self.states = np.concatenate((self.states, traj.obs), axis=0)
            self.actions = np.concatenate((self.actions, traj.actions), axis=0)
        else:
            self.states = np.concatenate((self.states, states), axis=0)
            self.actions = np.concatenate((self.actions, actions), axis=0)

    def traj_preprocess(self, traj):
        action_infos = traj.action_infos
        vals = np.array([ainfo['val'] for ainfo in action_infos])
        log_prob = np.array([ainfo['log_prob'] for ainfo in action_infos])
        adv = self.cal_advantages(traj)
        #TODO returns may not be discounted? 
        ret = adv + vals
        if cfg.alg.normalize_adv:
            adv = adv.astype(np.float64)
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        # data = dict(
        #     # ob=traj.obs,
        #     # action=traj.actions,
        #     ret=ret,
        #     adv=adv,
        #     log_prob=log_prob,
        #     val=vals
        # )

        return traj.obs, traj.actions, adv, ret, log_prob, vals

    #TODO migrate one of the cal_advantages? 

    def cal_advantages(self, traj):
        rewards = traj.rewards
        action_infos = traj.action_infos
        vals = np.array([ainfo['val'] for ainfo in action_infos])
        last_val = traj.extra_data['last_val']
        # there's also a torch version if necessary. 
        adv = cal_gae_torch(gamma=cfg.alg.rew_discount,
                        lam=cfg.alg.gae_lambda,
                        rewards=rewards,
                        value_estimates=vals,
                        last_value=last_val,
                        dones=traj.dones)
        return adv

    # from homework, probably will not use 
    def compute_advantage_gae(self, values, rewards, device=None):
        # values should be average values from the K-ensemble of critics
        advantages = torch.zeros_like(values, device=device)

        #### TODO: populate GAE in advantages over T timesteps (10 pts) ############

        # we do not have a future value for the last timestep
        for t in range(len(rewards) - 2, -1, -1):
            gae = rewards[t] + self.agent.discount * values[t + 1] - values[t]
            advantages[t] = gae + self.agent.gae_lambda * self.agent.discount * (advantages[t + 1])

        ############################################################################

        return advantages[:self.agent.T]




## Sample from homework>
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
                # logging from train_ppo
                det_log_info, _ = self.eval(eval_num=cfg.alg.test_num,
                                            sample=False, smooth=True)
                sto_log_info, _ = self.eval(eval_num=cfg.alg.test_num,
                                            sample=True, smooth=False)

                det_log_info = {f'det/{k}': v for k, v in det_log_info.items()}
                sto_log_info = {f'sto/{k}': v for k, v in sto_log_info.items()}
                eval_log_info = {**det_log_info, **sto_log_info}
                self.agent.save_model(is_best=self._eval_is_best,
                                      step=self.cur_step)
            else:
                eval_log_info = None
            # rollout the current policy and get a trajectory
            # TODO: rollout multiple trajectories
            # TODO: make num_traj exist somehow, add to config??? whatever this cfg.alg is 
            trajs, rollout_time = self.rollout_batch(cfg.alg.num_traj, sample=True,
                                                     get_last_val=True,
                                                     time_steps=cfg.alg.episode_steps)

            # TODO: got to train K times for each of the V values...
            train_log_infos = self.train_batch(trajs)
            # logging things
            if iter_t % cfg.alg.log_interval == 0:
                for train_log_info in train_log_infos:
                    train_log_info['train/rollout_time'] = rollout_time
                    if eval_log_info is not None:
                        train_log_info.update(eval_log_info)
                    if cfg.alg.linear_decay_lr:
                        train_log_info.update(self.agent.get_lr())
                    if cfg.alg.linear_decay_clip_range:
                        train_log_info.update(dict(clip_range=cfg.alg.clip_range))
                    scalar_log = {'scalar': train_log_info}
                    self.tf_logger.save_dict(scalar_log, step=self.cur_step)
            if self.cur_step > cfg.alg.max_steps:
                break
        return dataset_sizes, success_rates

    ## TODO ensure this eval works, it probably does not require change
    @torch.no_grad()
    def eval(self, render=False, save_eval_traj=False, eval_num=1,
             sleep_time=0, sample=True, smooth=True, no_tqdm=None):
        time_steps = []
        rets = []
        lst_step_infos = []
        successes = []
        if no_tqdm:
            disable_tqdm = bool(no_tqdm)
        else:
            disable_tqdm = not cfg.alg.test
        for idx in tqdm(range(eval_num), disable=disable_tqdm):
            traj, _ = self.rollout_once(time_steps=cfg.alg.episode_steps,
                                        return_on_done=True,
                                        sample=cfg.alg.sample_action and sample,
                                        render=render,
                                        sleep_time=sleep_time,
                                        render_image=save_eval_traj,
                                        evaluation=True)
            tsps = traj.steps_til_done.copy().tolist()
            rewards = traj.raw_rewards
            infos = traj.infos
            for ej in range(traj.num_envs):
                ret = np.sum(rewards[:tsps[ej], ej])
                rets.append(ret)
                lst_step_infos.append(infos[tsps[ej] - 1][ej])
            time_steps.extend(tsps)
            if save_eval_traj:
                save_traj(traj, cfg.alg.eval_dir)
            if 'success' in infos[0][0]:
                successes.extend([infos[tsps[ej] - 1][ej]['success'] for ej in range(rewards.shape[1])])

        raw_traj_info = {'return': rets,
                         'episode_length': time_steps,
                         'lst_step_info': lst_step_infos}
        log_info = dict()
        for key, val in raw_traj_info.items():
            if 'info' in key:
                continue
            val_stats = get_list_stats(val)
            for sk, sv in val_stats.items():
                log_info['eval/' + key + '/' + sk] = sv
        if len(successes) > 0:
            log_info['eval/success'] = np.mean(successes)
        if smooth:
            if self.smooth_eval_return is None:
                self.smooth_eval_return = log_info['eval/return/mean']
            else:
                self.smooth_eval_return = self.smooth_eval_return * self.smooth_tau
                self.smooth_eval_return += (1 - self.smooth_tau) * log_info['eval/return/mean']
            log_info['eval/smooth_return/mean'] = self.smooth_eval_return
            if self.smooth_eval_return > self._best_eval_ret:
                self._eval_is_best = True
                self._best_eval_ret = self.smooth_eval_return
            else:
                self._eval_is_best = False
        return log_info, raw_traj_info

    def train_batch(self, trajs, sample_percent=.9):
        # TODO: figure out how to calculate steps
        self.cur_step += torch.sum([traj.total_steps for traj in trajs])
        #moved preprocessing to inside the TrajDataset 
        self.dataset = TrajDataset(trajs)

        # TODO make sure we can iterate through the critics
        for critic in self.agent.critic:
            critic_optim_info = self.train_critic(trajs, critic)
            # consider logging critic optim info later?

        # TODO - ensure this optimizing is just for the actor, not value function
        rollout_dataloader = DataLoader(self.dataset,
                                        batch_size=cfg.alg.batch_size,
                                        shuffle=True,
                                        )
        optim_infos = []
        for oe in range(cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize_policy(batch_data)
                optim_infos.append(optim_info)
        return [self.get_train_log(optim_infos, traj) for traj in trajs]

    def train_critic(self, critic, sample_percent=.9):
        dataset_size = len(self.dataset.states)
        rand_indices = np.random.choice(dataset_size, size=np.floor(dataset_size * sample_percent))
        rollout_dataloader = DataLoader(Subset(self.dataset, rand_indices),
                                        batch_size=cfg.alg.batch_size,
                                        shuffle=True,
                                        )
        # TODO - update value function with rollout_dataloader stuff
        optim_infos = []
        for oe in range(cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize_value_function(batch_data)
                optim_infos.append(optim_info)
        return optim_infos

    def rollout_batch(self, num_trajs, **kwargs):
        t0 = time.perf_counter()
        self.agent.eval_mode()
        trajs = []
        for i in range(num_trajs):
            trajs.append(self.runner(**kwargs))
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        # trajs = torch.vstack(trajs)
        return trajs, elapsed_time

    def rollout_once(self, *args, **kwargs):
        t0 = time.perf_counter()
        self.agent.eval_mode()
        traj = self.runner(**kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        return traj, elapsed_time

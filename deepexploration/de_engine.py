import time
from collections import deque
from itertools import count
from dataclasses import dataclass

from easyrl.engine.basic_engine import BasicEngine
from easyrl.utils.rl_logger import TensorboardLogger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
from easyrl.utils.gae import cal_gae, cal_gae_torch
import numpy as np
import torch
import torch.optim as optim
from easyrl.configs import cfg
from tqdm.notebook import tqdm
from easyrl.utils.common import save_traj, get_list_stats
from typing import Any

from utils import eval_agent


class TrajDataset(Dataset):
    def __init__(self, traj):
        states = []
        actions = []
        values = []
        advantages = []
        returns = []
        log_probs = []

        obs, a, adv, ret, log_prob, val = self.traj_preprocess(traj)
        states.append(obs)
        actions.append(a)
        values.append(val)
        log_probs.append(log_prob)
        returns.append(ret)
        advantages.append(adv)

        self.data = {
            "ob": states,
            "action": actions,
            "val": values,
            "adv": advantages,
            "ret": returns,
            "log_prob": log_probs
        }
        self.states = states
        self.actions = actions

    def __len__(self):
        # return self.states.shape[0]
        return len(self.states)

    def __getitem__(self, idx):
        sample = dict()
        for key in self.data:
            sample[key] = self.data[key][idx]
        return sample

    def traj_preprocess(self, traj):
        action_infos = [step_data.action_info for step_data in traj.traj_data]
        vals = np.expand_dims(np.array([ainfo['val'] for ainfo in action_infos]), axis=1)
        log_prob = np.array([ainfo['log_prob'] for ainfo in action_infos])
        adv = self.cal_advantages(traj)
        ret = adv + vals.squeeze(1)
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

    # TODO migrate one of the cal_advantages?

    def cal_advantages(self, traj):
        rewards = np.array([step_data.reward for step_data in traj.traj_data])
        action_infos = [step_data.action_info for step_data in traj.traj_data]
        vals = np.array([ainfo['val'] for ainfo in action_infos])
        last_val = torch.from_numpy(traj.extra_data['last_val']).reshape(1)
        # there's also a torch version if necessary.
        adv = cal_gae(gamma=cfg.alg.rew_discount,
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

@dataclass
class DeepExplorationEngine(BasicEngine):
    agent: Any
    runner: Any
    env: Any
    smooth_eval_return: Any = None

    def __post_init__(self):
        self.cur_step = 0
        self._best_eval_ret = -np.inf
        self._eval_is_best = False
        if cfg.alg.test or cfg.alg.resume:
            self.cur_step = self.agent.load_model(step=cfg.alg.resume_step)
        else:
            if cfg.alg.pretrain_model is not None:
                self.agent.load_model(pretrain_model=cfg.alg.pretrain_model)
            cfg.alg.create_model_log_dir()
        self.train_ep_return = deque(maxlen=100)
        self.smooth_eval_return = None
        self.smooth_tau = cfg.alg.smooth_eval_tau
        self.optim_stime = None
        if not cfg.alg.test:
            self.tf_logger = TensorboardLogger(log_dir=cfg.alg.log_dir)

    # TODO: actually we can probably use the old train now, revert
    def train(self):
        success_rates = []
        self.cur_step = 0
        for iter_t in count():
            print(f'iter: {iter_t}')
            if iter_t % cfg.alg.eval_interval == 0:
                success_rate, ret_mean, ret_std, rets, successes = eval_agent(self.agent,
                                                                              self.env,
                                                                              1,
                                                                              disable_tqdm=True)
                success_rates.append(success_rate)
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

            traj, rollout_time = self.rollout_once(sample=True,
                                                   get_last_val=True,
                                                   time_steps=cfg.alg.episode_steps)
            train_log_info = self.train_once(traj)

            # logging things
            if iter_t % cfg.alg.log_interval == 0:
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
        return success_rates

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

    def train_once(self, traj):
        self.optim_stime = time.perf_counter()
        self.cur_step += traj.total_steps
        rollout_dataloader = TrajDataset(traj)
        optim_infos = []
        for oe in range(cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize(batch_data)
                optim_infos.append(optim_info)
        return self.get_train_log(optim_infos, traj)

    def rollout_once(self, *args, **kwargs):
        t0 = time.perf_counter()
        self.agent.eval_mode()
        traj = self.runner(**kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        return traj, elapsed_time

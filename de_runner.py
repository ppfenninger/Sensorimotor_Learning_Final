import numpy as np
from copy import deepcopy
from easyrl.utils.gym_util import get_true_done
from collections import deque
from easyrl.configs import cfg
import time
from copy import deepcopy

import numpy as np
import torch
from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.gym_util import get_render_images
from easyrl.utils.torch_util import torch_to_np


class BasicRunner:
    def __init__(self, agent, env, eval_env=None):
        self.agent = agent
        self.train_env = env
        self.num_train_envs = env.num_envs
        self.obs = None
        self.eval_env = env if eval_env is None else eval_env
        self.train_ep_return = deque(maxlen=cfg.alg.deque_size)
        self.train_ep_len = deque(maxlen=cfg.alg.deque_size)
        self.train_success = deque(maxlen=cfg.alg.deque_size)
        self.reset_record()

    def __call__(self, **kwargs):
        raise NotImplementedError

    def reset(self, env=None, *args, **kwargs):
        if env is None:
            env = self.train_env
        self.obs = env.reset(*args, **kwargs)
        self.reset_record()

    def reset_record(self):
        self.cur_ep_len = np.zeros(self.num_train_envs)
        self.cur_ep_return = np.zeros(self.num_train_envs)

    def get_true_done_next_ob(self, next_ob, done, reward, info, all_dones, skip_record=False):
        done_idx = np.argwhere(done).flatten()
        self.cur_ep_len += 1
        if 'raw_reward' in info[0]:
            self.cur_ep_return += np.array([x['raw_reward'] for x in info])
        else:
            self.cur_ep_return += reward
        if done_idx.size > 0:
            # vec env automatically resets the environment when it's done
            # so the returned next_ob is not actually the next observation
            true_next_ob = deepcopy(next_ob)
            true_next_ob[done_idx] = np.array([info[i]['true_next_ob'] for i in done_idx])
            if all_dones is not None:
                all_dones[done_idx] = True

            true_done = deepcopy(done)
            for iidx, inf in enumerate(info):
                true_done[iidx] = get_true_done(true_done[iidx], inf)
            if not skip_record:
                self.train_ep_return.extend([self.cur_ep_return[dix] for dix in done_idx])
                self.train_ep_len.extend([self.cur_ep_len[dix] for dix in done_idx])
                if 'success' in info[0]:
                    self.train_success.extend([info[i]['success'] for i in done_idx])
            self.cur_ep_return[done_idx] = 0
            self.cur_ep_len[done_idx] = 0
        else:
            true_next_ob = next_ob
            true_done = done
        return true_next_ob, true_done, all_dones

class EpisodicRunner(BasicRunner):
    """
    This only applies to environments that are wrapped by VecEnv.
    It assumes the environment is automatically reset if done=True
    """

    def __init__(self, *args, **kwargs):
        super(EpisodicRunner, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, reset_first=False,
                 reset_kwargs=None, action_kwargs=None,
                 random_action=False, get_last_val=False):
        traj = Trajectory()
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        if self.obs is None or reset_first or evaluation:
            self.reset(env=env, **reset_kwargs)
        ob = self.obs
        # this is critical for some environments depending
        # on the returned ob data. use deepcopy() to avoid
        # adding the same ob to the traj

        # only add deepcopy() when a new ob is generated
        # so that traj[t].next_ob is still the same instance as traj[t+1].ob
        ob = deepcopy(ob)
        if return_on_done:
            all_dones = np.zeros(env.num_envs, dtype=bool)
        else:
            all_dones = None
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = get_render_images(env)

            if random_action:
                action = env.random_actions()
                action_info = dict()
            else:
                action, action_info = self.agent.get_action(ob,
                                                            sample=sample,
                                                            **action_kwargs)
            next_ob, reward, done, info = env.step(action)

            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            true_next_ob, true_done, all_dones = self.get_true_done_next_ob(next_ob,
                                                                            done,
                                                                            reward,
                                                                            info,
                                                                            all_dones,
                                                                            skip_record=evaluation)
            sd = StepData(ob=ob,
                          action=action,
                          action_info=action_info,
                          next_ob=true_next_ob,
                          reward=reward,
                          done=true_done,
                          info=info)
            ob = next_ob
            traj.add(sd)
            if return_on_done and np.all(all_dones):
                break

        if get_last_val and not evaluation:
            last_val = self.agent.get_val(traj[-1].next_ob)
            traj.add_extra('last_val', torch_to_np(last_val))
        self.obs = ob if not evaluation else None
        return traj

class DeepExplorationRunner(BasicRunner):
    '''Creates a new runner with epsilon greedy steps to roll out trajectories'''

    def __init__(self, *args, **kwargs):
        super(DeepExplorationRunner, self).__init__(*args, **kwargs)
        self.epsilon = 0.1
        self.num_steps = 10
        self.num_envs = self.train_env.num_envs
        self.num_actions = self.train_env.action_space.n

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        traj = Trajectory()
        if self.obs is None:
            self.reset()
        ob = self.obs
        for t in range(self.num_steps):
            action = np.random.randint(self.num_actions, size=self.num_envs)
            action_info = dict()
            next_ob, reward, done, info = self.train_env.step(action)
            true_next_ob, true_done, _ = self.get_true_done_next_ob(next_ob,
                                                                     done,
                                                                     reward,
                                                                     info,
                                                                     None,
                                                                     skip_record=True)
            sd = StepData(ob=ob,
                          action=action,
                          action_info=action_info,
                          next_ob=true_next_ob,
                          reward=reward,
                          done=true_done,
                          info=info)
            ob = next_ob
            traj.add(sd)
        self.obs = ob
        return traj
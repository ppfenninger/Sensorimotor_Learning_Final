from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs import cfg
from easyrl.utils.common import linear_decay_percent
from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import clip_grad
from easyrl.utils.torch_util import load_ckpt_data
from easyrl.utils.torch_util import load_state_dict
from easyrl.utils.torch_util import move_to
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np
from typing import List


@dataclass
class DeepExplorationAgent(BaseAgent):
    actor: nn.Module
    critics: List[nn.Module]
    same_body: float = False
    is_in_exploration_mode = False
    exploration_horizon = 5
    exploration_steps = 0
    beta = 0.1
    epsilon = 0.2
    k_samples = 10
    gae_lambda = .5 #will need to move to config 
    discount = .9 #will need to move to config 

    def __post_init__(self):
        move_to([self.actor] + self.critics,
                device=cfg.alg.device)
        if cfg.alg.vf_loss_type == 'mse':
            self.val_loss_criterion = nn.MSELoss().to(cfg.alg.device)
        elif cfg.alg.vf_loss_type == 'smoothl1':
            self.val_loss_criterion = nn.SmoothL1Loss().to(cfg.alg.device)
        else:
            raise TypeError(f'Unknown value loss type: {cfg.alg.vf_loss_type}!')
        all_params = list(self.actor.parameters())
        for critic in self.critics:
            all_params += list(critic.parameters())
        # keep unique elements only. The following code works for python >=3.7
        # for earlier version of python, u need to use OrderedDict
        self.all_params = dict.fromkeys(all_params).keys()
        if (cfg.alg.linear_decay_lr or cfg.alg.linear_decay_clip_range) and \
                cfg.alg.max_steps > cfg.alg.max_decay_steps:
            logger.warning('max_steps should not be greater than max_decay_steps.')
            cfg.alg.max_decay_steps = int(cfg.alg.max_steps * 1.5)
            logger.warning(f'Resetting max_decay_steps to {cfg.alg.max_decay_steps}!')
        total_epochs = int(np.ceil(cfg.alg.max_decay_steps / (cfg.alg.num_envs *
                                                              cfg.alg.episode_steps)))
        if cfg.alg.linear_decay_clip_range:
            self.clip_range_decay_rate = cfg.alg.clip_range / float(total_epochs)

        p_lr_lambda = partial(linear_decay_percent,
                              total_epochs=total_epochs)
        optim_args = dict(
            lr=cfg.alg.policy_lr,
            weight_decay=cfg.alg.weight_decay
        )
        if not cfg.alg.sgd:
            optim_args['amsgrad'] = cfg.alg.use_amsgrad
            optim_func = optim.Adam
        else:
            optim_args['nesterov'] = True if cfg.alg.momentum > 0 else False
            optim_args['momentum'] = cfg.alg.momentum
            optim_func = optim.SGD
        if self.same_body:
            optim_args['params'] = self.all_params
        else:
            optim_args['params'] = [{'params': self.actor.parameters(),
                                     'lr': cfg.alg.policy_lr},
                                    {'params': self.critics[0].parameters(), #TODO: fix critics
                                     'lr': cfg.alg.value_lr}]

        self.optimizer = optim_func(**optim_args)

        if self.same_body:
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                         lr_lambda=[p_lr_lambda])
        else:
            v_lr_lambda = partial(linear_decay_percent,
                                  total_epochs=total_epochs)
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                         lr_lambda=[p_lr_lambda, v_lr_lambda])

    @torch.no_grad()
    def get_candidate_actions(self, ob, sample=True, *args, **kwargs):
        '''When in exploration mode, we call this function to explore actions and select
        which action to take. We explore k_samples actions and return them. The DE Runner will take these actions
         and evaluate what the next state will be. We can think of this as having a forward model which is a copy
         of our enviornment which the runner has access to. We can imagine a world where instead of actually simulating
         each candidate action on our real environment, we simulate them on this forward model. This is the idea behind
         model based RL.'''
        self.eval_mode()
        t_ob = torch_float(ob, device=cfg.alg.device)
        act_dist, avg_val, std_val = self.get_act_val_ensemble_stats(t_ob)

        def get_action_info(action, func_act_dist):
            log_prob = action_log_prob(action, func_act_dist)
            entropy = action_entropy(func_act_dist, log_prob)
            return dict(
                log_prob=torch_to_np(log_prob),
                entropy=torch_to_np(entropy),
        )

        candidate_actions = [action_from_dist(act_dist, sample=sample) for _ in range(self.k_samples)]
        candidate_action_infos = [get_action_info(action, act_dist) for action in candidate_actions]
        return candidate_actions, candidate_action_infos

    @torch.no_grad()
    def get_candidate_scores(self, next_states_from_candidate_actions, *args, **kwargs):
        '''This function takes in the next states from the candidate actions (using our environment but we could also
        use a forward model) and returns scores for each candidate action. These scores are the sum of the average
        value of the next state from our critics and beta times the standard deviation of the next state from our
        critics.'''
        self.eval_mode()
        avg_values = []
        std_values = []
        for next_state in next_states_from_candidate_actions:
            act_dist, avg_val, std_val = self.get_act_val_ensemble_stats(next_state)
            avg_values.append(avg_val)
            std_values.append(std_val)

        scores = [avg_val + self.beta * std_val for avg_val, std_val in zip(avg_values, std_values)]
        return scores

    @torch.no_grad()
    def get_exploration_bonus(self, state):
        '''This function returns the exploration bonus for a given state. This is beta times the std deviation of the values at
        that state according to the ensemble of critics.'''
        self.eval_mode()
        act_dist, avg_val, std_val = self.get_act_val_ensemble_stats(state)
        return self.beta * std_val

    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        self.eval_mode()
        t_ob = torch_float(ob, device=cfg.alg.device)
        act_dist, avg_val, std_val = self.get_act_val_ensemble_stats(t_ob)
        action = action_from_dist(act_dist,
                                  sample=sample)
        log_prob = action_log_prob(action, act_dist)
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy)
        )

        return torch_to_np(action), action_info

    @torch.no_grad()
    def get_act_val_ensemble_stats(self, ob, *args, **kwargs):
        '''Returns the action distribution, the average value of the critics, and
        the std of the critics'''
        act_dist, vals = self.get_act_vals_from_ensemble(ob, *args, **kwargs)
        return act_dist, torch.mean(vals), torch.std(vals)

    @torch.no_grad()
    def get_act_vals_from_ensemble(self, ob, *args, **kwargs):
        '''Returns the action distribution and values from all the critics'''
        ob = torch_float(ob, device=cfg.alg.device)
        act_dist, body_out = self.actor(ob)
        # vals = torch.tensor([critic(x=ob)[0].squeeze(-1) for critic in self.critics])
        vals = torch.tensor([self.get_act_val(ob, i) for i in range(len(self.critics))])
        return act_dist, vals

    @torch.no_grad()
    def get_act_val(self, ob, critic_index, *args, **kwargs):
        '''
        Returns the value of the critic at critic_index predicts for ob
        '''
        self.eval_mode()
        ob = torch_float(ob, device=cfg.alg.device)
        val, body_out = self.critics[critic_index](x=ob)
        val = val.squeeze(-1)
        return val

    def optimize(self, data, *args, **kwargs):
        processed_data = self.optim_preprocess(data)
        
        loss, pg_loss, vf_loss = self.cal_loss(**processed_data)
        
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = clip_grad(self.all_params, cfg.alg.max_grad_norm)
        self.optimizer.step()

        optim_info = dict(
            pg_loss=pg_loss.item(),
            vf_loss=vf_loss.item(),
            total_loss=loss.item(),
        )
        optim_info['grad_norm'] = grad_norm
        return optim_info
         

    def optim_preprocess(self, data):
        self.train_mode()
        for key, val in data.items():
            data[key] = torch_float(val, device=cfg.alg.device)
        ob = data['ob']
        action = data['action']
        ret = data['ret']
        adv = data['adv']
        # old_val = data['val']

        act_dist, vals  = self.get_act_vals_from_ensemble(ob)
        log_prob = action_log_prob(action, act_dist)
        # entropy = action_entropy(act_dist, log_prob)
        if not all([x.ndim == 1 for x in [log_prob, vals[0]]]):
            raise ValueError('val, log_prob should be 1-dim!')
        processed_data = dict(
            ob=ob,
            vals=vals, # this is a list of vals for each critic 
            ret=ret,
            log_prob=log_prob,
            adv=adv,

        )
        return processed_data

    def cal_loss(self, ob, vals, ret, log_prob, adv):

        rand_indices = np.random.choice(len(self.critics), size=np.floor(len(self.critics) * cfg.alg.sample_percent))
        # rand_indices = torch.cuda.FloatTensor(10, 10).uniform_() > 0.8
        vf_loss = self.cal_val_loss(ob=ob, vals=vals, ret=ret)
        pg_loss = -torch.mean(log_prob*adv, axis=-1) 
        loss = pg_loss + vf_loss * cfg.alg.vf_coef
        return loss, pg_loss, vf_loss
    
    def cal_val_loss(self, ob, vals, ret, rand_indices=True):
        # TODO check what the output of the critic actually is 
        # val = torch.mean([self.critics[i](ob) for i in critic_indices])
        ensemble_mask = 1
        if rand_indices: 
            ensemble_mask = torch.cuda.FloatTensor(10, 10).uniform_() > 0.8
        val = torch.mean(ensemble_mask*vals)
        vf_loss = F.mse_loss( val, ret )
        return vf_loss

    def train_mode(self):
        self.actor.train()
        for critic in self.critics:
            critic.train()

    def eval_mode(self): #doesn't collect gradients
        self.actor.eval()
        for critic in self.critics:
            critic.eval()

    def decay_lr(self):
        self.lr_scheduler.step()

    def get_lr(self):
        cur_lr = self.lr_scheduler.get_lr()
        lrs = {'policy_lr': cur_lr[0]}
        if len(cur_lr) > 1:
            lrs['value_lr'] = cur_lr[1]
        return lrs

    def decay_clip_range(self):
        cfg.alg.clip_range -= self.clip_range_decay_rate

    def save_model(self, is_best=False, step=None):
        self.save_env(cfg.alg.model_dir)
        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }

        for i, module in enumerate(self.critics):
            data_to_save[f'critic_state_dict_{i}'] = module.state_dict()

        if cfg.alg.linear_decay_clip_range:
            data_to_save['clip_range'] = cfg.alg.clip_range
            data_to_save['clip_range_decay_rate'] = self.clip_range_decay_rate
        save_model(data_to_save, cfg.alg, is_best=is_best, step=step)

    def load_model(self, step=None, pretrain_model=None):
        self.load_env(cfg.alg.model_dir)
        ckpt_data = load_ckpt_data(cfg.alg, step=step,
                                   pretrain_model=pretrain_model)
        load_state_dict(self.actor,
                        ckpt_data['actor_state_dict'])
        for i, critic in enumerate(self.critics):
            load_state_dict(critic,
                            ckpt_data[f'critic_state_dict_{i}'])
        if pretrain_model is not None:
            return
        self.optimizer.load_state_dict(ckpt_data['optim_state_dict'])
        self.lr_scheduler.load_state_dict(ckpt_data['lr_scheduler_state_dict'])
        if cfg.alg.linear_decay_clip_range:
            self.clip_range_decay_rate = ckpt_data['clip_range_decay_rate']
            cfg.alg.clip_range = ckpt_data['clip_range']
        return ckpt_data['step']

    def print_param_grad_status(self):
        logger.info('Requires Grad?')
        logger.info('================== Actor ================== ')
        for name, param in self.actor.named_parameters():
            print(f'{name}: {param.requires_grad}')
        logger.info('================== Critic ================== ')
        for name, param in self.critics[0].named_parameters(): #TODO fix critics
            print(f'{name}: {param.requires_grad}')


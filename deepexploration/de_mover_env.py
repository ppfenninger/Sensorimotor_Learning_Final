import gym
import numpy as np
import torch
import airobot as ar
from airobot.utils.common import euler2quat
from easyrl.utils.torch_util import torch_to_np
from gym import spaces


class URRobotGym(gym.Env):
    def __init__(self,
                 action_repeat=10,
                 use_sparse_reward=False,
                 use_subgoal=False,
                 with_obstacle=True,
                 apply_collision_penalty=False,
                 # Set 'gui' to False if you are using Colab, otherwise the session will crash as Colab does not support X window
                 # You can set it to True for debugging purpose if you are running the notebook on a local machine.
                 gui=False,
                 is_exploratory=False,
                 max_episode_length=25,
                 dist_threshold=0.05):
        self._action_repeat = action_repeat
        self._max_episode_length = max_episode_length
        self._dist_threshold = dist_threshold
        self._use_sparse_reward = use_sparse_reward
        self._use_subgoal = use_subgoal
        self._apply_collision_penalty = apply_collision_penalty
        self._with_obstacle = with_obstacle
        self.is_exploratory = is_exploratory
        print(f'================================================')
        print(f'Use sparse reward:{self._use_sparse_reward}')
        print(f'Use subgoal:{self._use_subgoal}')
        print(f'With obstacle in the scene:{self._with_obstacle}')
        print(f'Apply collision penalty:{self._apply_collision_penalty}')
        print(f'================================================')

        self._xy_bounds = np.array([[0.23, 0.78],  # [xmin, xmax]
                                    [-0.35, 0.3]])  # [ymin, ymax]
        self.robot = ar.Robot('ur5e_stick',
                              pb_cfg={'gui': gui,
                                   'realtime': False,
                                   'opengl_render': False})
        self._arm_reset_pos = np.array([-0.38337763,
                                        -2.02650575,
                                        -2.01989619,
                                        -0.64477803,
                                        1.571439041,
                                        -0.38331266])
        self._table_id = self.robot.pb_client.load_urdf('table/table.urdf',
                                                        [.5, 0, 0.4],
                                                        euler2quat([0, 0, np.pi / 2]),
                                                        scaling=0.9)

        # create a ball at the start location (for visualization purpose)
        self._start_pos = np.array([0.45, -0.32, 1.0])
        self._start_urdf_id = self.robot.pb_client.load_geom('sphere', size=0.04, mass=0,
                                                             base_pos=self._start_pos,
                                                             rgba=[1, 1, 0, 0.8])

        # create a ball at the goal location
        self._goal_pos = np.array([0.5, 0.26, 1.0])
        self._goal_urdf_id = self.robot.pb_client.load_geom('sphere', size=0.04, mass=0,
                                                            base_pos=self._goal_pos,
                                                            rgba=[1, 0, 0, 0.8])

        # disable the collision checking between the robot and the ball at the goal location
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            self.robot.pb_client.setCollisionFilterPair(self.robot.arm.robot_id,
                                                        self._goal_urdf_id,
                                                        i,
                                                        -1,
                                                        enableCollision=0)
        # disable the collision checking between the robot and the ball at the start location
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            self.robot.pb_client.setCollisionFilterPair(self.robot.arm.robot_id,
                                                        self._start_urdf_id,
                                                        i,
                                                        -1,
                                                        enableCollision=0)

        # create an obstacle
        if self._with_obstacle:
            self._wall_id = self.robot.pb_client.load_geom('box', size=[0.18, 0.01, 0.1], mass=0,
                                                           base_pos=[0.5, 0.15, 1.0],
                                                           rgba=[0.5, 0.5, 0.5, 0.8])

        # create balls at subgoal locations
        if self._use_subgoal:
            self._subgoal_pos = np.array([[0.24, 0.15, 1.0], [0.76, 0.15, 1.0]])
            self._subgoal_urdf_id = []
            for pos in self._subgoal_pos:
                self._subgoal_urdf_id.append(self.robot.pb_client.load_geom('sphere', size=0.04, mass=0,
                                                                            base_pos=pos,
                                                                            rgba=[0, 0.8, 0.8, 0.8]))
            # disable the collision checking between the robot and the subgoal balls
            for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
                for sg in self._subgoal_urdf_id:
                    self.robot.pb_client.setCollisionFilterPair(self.robot.arm.robot_id,
                                                                sg,
                                                                i,
                                                                -1,
                                                                enableCollision=0)
        self._subgoal_reached = False

        self._action_bound = 1.0
        self._ee_pos_scale = 0.02
        self._action_high = np.array([self._action_bound] * 2)
        self.action_space = spaces.Box(low=-self._action_high,
                                       high=self._action_high,
                                       dtype=np.float32)
        state_low = np.full(len(self._get_obs()), -float('inf'))
        state_high = np.full(len(self._get_obs()), float('inf'))
        self.observation_space = spaces.Box(state_low,
                                            state_high,
                                            dtype=np.float32)
        self.reset()

    def reset(self):
        self.robot.arm.set_jpos(self._arm_reset_pos, ignore_physics=True)
        self._t = 0
        self._ref_ee_pos = self.robot.arm.get_ee_pose()[0]
        self._ref_ee_ori = self.robot.arm.get_ee_pose()[1]
        return self._get_obs()

    def step(self, action):
        if hasattr(action[0], "__len__"):
            is_exploratory = action[1]
            action = action[0][0]
        else:
            is_exploratory = False
        if is_exploratory:
            state, reward, done, info = self._simulate_step(action)
        else:
            collision = self._apply_action(action)
            self._t += 1
            state = self._get_obs()
            done = self._t >= self._max_episode_length
            reward, info = self._get_reward(state=state, action=action, collision=float(collision))
            info['collision'] = collision
        return state, reward, done, info

    def _simulate_step(self, action):
        previous_arm_position = self.robot.arm.get_jpos()
        previous_arm_velocity = self.robot.arm.get_jvel()

        collision = self._apply_action(action)
        state = self._get_obs()

        reward, info = self._get_reward(state=state, action=action, collision=collision)
        done = self._t >= self._max_episode_length or info['success']
        info['collision'] = collision

        # reset the robot
        self.robot.arm.set_jpos(previous_arm_position, ignore_physics=True)
        self.robot.arm.set_jvel(previous_arm_velocity) #TODO: set with velocities

        self._ref_ee_pos = self.robot.arm.get_ee_pose()[0]
        self._ref_ee_ori = self.robot.arm.get_ee_pose()[1]

        return state, reward, done, info

    def seed(self, seed_i):
        from gym.utils import seeding
        self._np_random, _ = seeding.np_random(seed_i)

    def _get_reward(self, state, action, collision):
        dist_to_goal = np.linalg.norm(state - self._goal_pos[:2])
        success = dist_to_goal < self._dist_threshold
        reward = -dist_to_goal

        # if self._apply_collision_penalty:
        #     #### TODO: Q4 apply a collision penalty
        #     if collision:
        #         reward = -1

        info = dict(success=success)
        return reward, info

    def _get_reward_with_subgoal(self, state):
        reward = 0
        #### TODO: Q5 design a reward based on the state, goal and subgoal positions
        dists_to_subgoal = []
        for subgoal in self._subgoal_pos:
            dists_to_subgoal.append(np.linalg.norm(state - subgoal[:2]))

        reward = -min(dists_to_subgoal);
        if min(dists_to_subgoal) < self._dist_threshold:
            self._use_subgoal = False

        return reward

    def _get_obs(self):
        gripper_pos = self.robot.arm.get_ee_pose()[0][:2]
        state = gripper_pos
        return state

    def _check_collision_with_wall(self):
        if hasattr(self, '_wall_id'):
            return len(self.robot.pb_client.getContactPoints(self.robot.arm.robot_id,
                                                             self._wall_id, 10, -1)) > 0
        else:
            return False

    def _apply_action(self, action):
        jnt_poses = self.robot.arm.get_jpos()
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 2:
            raise ValueError('Action should be [d_x, d_y].')
        # we set dz=0
        action = np.append(action, 0)
        pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
        pos += action[:3] * self._ee_pos_scale
        pos[2] = self._ref_ee_pos[2]
        # if the new position is out of the bounds, then we don't apply the action
        if not np.logical_and(np.all(pos[:2] >= self._xy_bounds[:, 0]),
                              np.all(pos[:2] <= self._xy_bounds[:, 1])):
            return False

        # move the end-effector to the new position
        jnt_pos = self.robot.arm.compute_ik(pos, ori=self._ref_ee_ori)
        for step in range(self._action_repeat):
            self.robot.arm.set_jpos(jnt_pos)
            self.robot.pb_client.stepSimulation()

        # if collision occurs, we reset the robot back to its original pose (before apply_action)
        collision = False
        if self._check_collision_with_wall():
            self.robot.arm.set_jpos(jnt_poses, ignore_physics=True)
            collision = True
        return collision

    def render(self, mode, **kwargs):
        robot_base = self.robot.arm.robot_base_pos
        self.robot.cam.setup_camera(focus_pt=robot_base,
                                    dist=2,
                                    yaw=85,
                                    pitch=-20,
                                    roll=0)
        rgb, _ = self.robot.cam.get_images(get_rgb=True,
                                           get_depth=False)
        return rgb
import isaacgym
assert isaacgym
import torch
import gym

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, reward_scaling=1.0):
        super().__init__(env)
        self.env = env
        self.reward_scaling = reward_scaling

        self.obs_history_length = self.env.cfg.env.num_observation_history
        self.history_frame_skip = self.env.cfg.env.history_frame_skip

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history_buf = torch.zeros(self.env.num_envs, self.obs_history_length * self.history_frame_skip, self.num_obs, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

        self.reward_container.load_env(self)
        
    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.obs_history = self.obs_history_buf[:, self.history_frame_skip-1::self.history_frame_skip, :].reshape(self.env.num_envs, -1)
        assert self.obs_history[:, -self.num_obs:].allclose(obs[:, :]), "obs_history does not end with obs"
        
        env_ids = self.env.reset_buf.nonzero(as_tuple=False).flatten()
        self.obs_history_buf[env_ids, :, :] = 0
        self.obs_history[env_ids, :] = 0
        
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew * self.reward_scaling, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.obs_history = self.obs_history_buf[:, self.history_frame_skip-1::self.history_frame_skip, :].reshape(self.env.num_envs, -1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history_buf[env_ids, :, :] = 0
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

import time
import random
import shutil
from collections import deque
from datetime import timedelta

import torch
import numpy as np
# from ml_logger import logger

from params_proto import PrefixProto

from wbc_compliance_rl.modules.actor_critic import ActorCritic
from wbc_compliance_rl.logging.experiment_logger import ExperimentLogger
from wbc_compliance_rl.storage.rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from wbc_compliance_rl.utils.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24  # per iteration
    max_iterations = 5000  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 0
    log_freq = 1

    # load and resume
    resume = False
    resume_supercloud = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    resume_checkpoint = 'ac_weights_last.pt'



class OnPolicyRunner:

    def __init__(self, env, device='cpu', task_name='default', run_name='run', log_config=None,
                 wandb_init_kwargs=None, train_cfg=None):
        from wbc_compliance_rl.algorithms.ppo_cse import PPO

        self.device = device
        self.env = env
        self.train_cfg = train_cfg
        self.log_config = log_config
        self.runner_cfg = train_cfg.runner if train_cfg is not None else RunnerArgs
        policy_cfg = train_cfg.policy if train_cfg is not None else None
        algorithm_cfg = train_cfg.algorithm if train_cfg is not None else None

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      cfg=policy_cfg,
                                      ).to(self.device)

        from wbc_compliance_gym import WBC_COMPLIANCE_ROOT_DIR
        resume_checkpoint_path = None
        if self.runner_cfg.resume:
            import wandb
            body = wandb.restore(self.runner_cfg.resume_checkpoint, run_path=self.runner_cfg.resume_path)
            resume_checkpoint_path = body.name
        elif self.runner_cfg.resume_supercloud:
            print(f"Loading weights from checkpoint ({self.runner_cfg.resume_checkpoint}) and run path ({self.runner_cfg.resume_path} and {WBC_COMPLIANCE_ROOT_DIR})")
            resume_checkpoint_path = (
                WBC_COMPLIANCE_ROOT_DIR + "/resume_runs/" + self.runner_cfg.resume_path
                + "/" + self.runner_cfg.resume_checkpoint
            )
            print("path: ", resume_checkpoint_path)

        self.alg = PPO(actor_critic, device=self.device, cfg=algorithm_cfg)
        self.num_steps_per_env = self.runner_cfg.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -self.runner_cfg.save_video_interval

        if resume_checkpoint_path is not None:
            self.load(resume_checkpoint_path)
            print(
                f"Successfully loaded checkpoint ({self.runner_cfg.resume_checkpoint}) "
                f"from ({self.runner_cfg.resume_path})"
            )

        self.logger = ExperimentLogger(
            task_name=task_name,
            run_name=run_name,
            config=log_config,
            wandb_init_kwargs=wandb_init_kwargs,
        )
        self.last_saved_iteration = None

        self.env.reset()

    @staticmethod
    def _mean_episode_metrics(episode_infos):
        averaged = {}
        keys = {key for info in episode_infos for key in info}
        for key in keys:
            values = [info[key] for info in episode_infos if key in info]
            tensor_values = [value.reshape(()) for value in values
                             if isinstance(value, torch.Tensor) and value.numel() == 1]
            numeric_values = [value for value in values if isinstance(value, (int, float))]
            if tensor_values:
                averaged[key] = torch.stack(tensor_values).mean()
            elif numeric_values:
                averaged[key] = sum(numeric_values) / len(numeric_values)
        return averaged

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.logger.watch(self.alg.actor_critic, log_freq=self.runner_cfg.log_freq)

        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        num_train_envs = self.env.num_train_envs
        obs_dict = self.env.get_observations()
        obs = obs_dict["obs"].to(self.device)
        privileged_obs = obs_dict["privileged_obs"].to(self.device)
        obs_history = obs_dict["obs_history"].to(self.device)
        self.alg.actor_critic.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(num_train_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(num_train_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            iteration_start = time.time()
            episode_infos = []
            completed_rewards = []
            completed_lengths = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(
                        obs[:num_train_envs], privileged_obs[:num_train_envs], obs_history[:num_train_envs]
                    )
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs = obs_dict["obs"].to(self.device)
                    privileged_obs = obs_dict["privileged_obs"].to(self.device)
                    obs_history = obs_dict["obs_history"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if "train/episode" in infos:
                        episode_infos.append(infos["train/episode"])

                    cur_reward_sum += rewards[:num_train_envs]
                    cur_episode_length += 1
                    done_ids = (dones[:num_train_envs] > 0).nonzero(as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        completed_rewards.append(cur_reward_sum[done_ids])
                        completed_lengths.append(cur_episode_length[done_ids])
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

            collection_time = time.time() - iteration_start
            learning_start = time.time()
            update_result = self.alg.update()
            learning_time = time.time() - learning_start
            iteration_time = time.time() - iteration_start

            (mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss,
             mean_decoder_loss, mean_decoder_loss_student,
             mean_adaptation_module_test_loss, mean_decoder_test_loss,
             mean_decoder_test_loss_student, mean_adaptation_losses_dict) = update_result

            if completed_rewards:
                rewbuffer.extend(torch.cat(completed_rewards).cpu().tolist())
                lenbuffer.extend(torch.cat(completed_lengths).cpu().tolist())

            iteration_timesteps = self.num_steps_per_env * num_train_envs
            self.tot_timesteps += iteration_timesteps
            self.tot_time += iteration_time
            fps = int(iteration_timesteps / max(iteration_time, 1e-6))
            mean_reward = sum(rewbuffer) / len(rewbuffer) if rewbuffer else 0.0
            mean_episode_length = sum(lenbuffer) / len(lenbuffer) if lenbuffer else 0.0
            mean_noise_std = self.alg.actor_critic.std.mean().detach()

            metrics = {
                "Loss/value_function": mean_value_loss,
                "Loss/surrogate": mean_surrogate_loss,
                "Loss/adaptation": mean_adaptation_module_loss,
                "Loss/decoder": mean_decoder_loss,
                "Loss/decoder_student": mean_decoder_loss_student,
                "Loss/adaptation_test": mean_adaptation_module_test_loss,
                "Loss/decoder_test": mean_decoder_test_loss,
                "Loss/decoder_test_student": mean_decoder_test_loss_student,
                "Loss/learning_rate": self.alg.learning_rate,
                "Policy/mean_noise_std": mean_noise_std,
                "Train/mean_reward": mean_reward,
                "Train/mean_episode_length": mean_episode_length,
                "Perf/collection_time": collection_time,
                "Perf/learning_time": learning_time,
                "Perf/iteration_time": iteration_time,
                "Perf/fps": fps,
                "Train/total_timesteps": self.tot_timesteps,
                "Train/iteration": it,
            }
            metrics.update({f"Episode/{key}": value for key, value in
                            self._mean_episode_metrics(episode_infos).items()})
            metrics.update({f"Adaptation/{key}": value
                            for key, value in mean_adaptation_losses_dict.items()})
            scalar_metrics = self.logger.log(metrics, step=it)

            eta_seconds = iteration_time * (tot_iter - it - 1)
            self._print_iteration(
                it, tot_iter, fps, collection_time, learning_time,
                scalar_metrics, eta_seconds,
            )

            if self.runner_cfg.save_video_interval:
                self.log_video(it)

            if self.runner_cfg.save_interval > 0 and it % self.runner_cfg.save_interval == 0:
                self.save(
                    self.logger.checkpoint_dir / f"model_{it:06d}.pt",
                    iteration=it,
                )

            self.current_learning_iteration = it + 1

        self.save(
            self.logger.checkpoint_dir /
            f"model_{self.current_learning_iteration:06d}.pt",
            iteration=self.current_learning_iteration,
        )
        self.logger.close()

    def _print_iteration(self, it, tot_iter, fps, collection_time, learning_time,
                         metrics, eta_seconds):
        width = 80
        title = f" Learning iteration {it}/{tot_iter - 1} "
        print("\n" + title.center(width, "#"))
        print(f"{'Computation:':>35} {fps} steps/s")
        print(f"{'Collection time:':>35} {collection_time:.3f} s")
        print(f"{'Learning time:':>35} {learning_time:.3f} s")
        print(f"{'Value function loss:':>35} {metrics['Loss/value_function']:.4f}")
        print(f"{'Surrogate loss:':>35} {metrics['Loss/surrogate']:.4f}")
        print(f"{'Mean action noise std:':>35} {metrics['Policy/mean_noise_std']:.4f}")
        print(f"{'Mean reward:':>35} {metrics['Train/mean_reward']:.2f}")
        print(f"{'Mean episode length:':>35} {metrics['Train/mean_episode_length']:.2f}")
        print("-" * width)
        print(f"{'Total timesteps:':>35} {self.tot_timesteps}")
        print(f"{'Iteration time:':>35} {metrics['Perf/iteration_time']:.2f} s")
        print(f"{'Total time:':>35} {self.tot_time:.2f} s")
        print(f"{'ETA:':>35} {timedelta(seconds=int(max(0, eta_seconds)))}")
        print("#" * width, flush=True)

    def save(self, path, iteration=None, infos=None):
        from wbc_compliance_rl.utils.policy_export import export_policy_as_jit

        path = str(path)
        if iteration is None:
            iteration = self.current_learning_iteration
        checkpoint = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "adaptation_module_optimizer_state_dict":
                self.alg.adaptation_module_optimizer.state_dict(),
            "iter": iteration,
            "infos": infos,
            "config": self.log_config,
            "runner_state": {
                "tot_timesteps": self.tot_timesteps,
                "tot_time": self.tot_time,
                "learning_rate": self.alg.learning_rate,
            },
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
        if hasattr(self.alg, "decoder_optimizer"):
            checkpoint["decoder_optimizer_state_dict"] = (
                self.alg.decoder_optimizer.state_dict()
            )
        torch.save(checkpoint, path)
        latest_path = str(self.logger.checkpoint_dir / "model_latest.pt")
        if str(path) != latest_path:
            shutil.copy2(path, latest_path)
        self.logger.save(path)

        policy_dir = self.logger.export_dir / "policies"
        export_path = export_policy_as_jit(
            self.alg.actor_critic,
            policy_dir,
            filename=f"policy_{iteration:06d}.pt",
        )
        latest_export_path = policy_dir / "policy_latest.pt"
        shutil.copy2(export_path, latest_export_path)
        self.logger.save(export_path)
        self.last_saved_iteration = iteration
        print(f"Saved checkpoint {iteration}: {path}", flush=True)
        print(f"Exported policy {iteration}: {export_path}", flush=True)

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        # Compatibility with checkpoints produced before the runner-aligned
        # checkpoint format was introduced.
        if "model_state_dict" not in loaded:
            self.alg.actor_critic.load_state_dict(loaded)
            return None

        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        if load_optimizer:
            optimizer_state = loaded.get("optimizer_state_dict")
            if optimizer_state is not None:
                self.alg.optimizer.load_state_dict(optimizer_state)
            adaptation_state = loaded.get("adaptation_module_optimizer_state_dict")
            if adaptation_state is not None:
                self.alg.adaptation_module_optimizer.load_state_dict(adaptation_state)
            decoder_state = loaded.get("decoder_optimizer_state_dict")
            if decoder_state is not None and hasattr(self.alg, "decoder_optimizer"):
                self.alg.decoder_optimizer.load_state_dict(decoder_state)
        self.current_learning_iteration = int(loaded.get("iter", 0))
        runner_state = loaded.get("runner_state", {})
        self.tot_timesteps = int(runner_state.get("tot_timesteps", 0))
        self.tot_time = float(runner_state.get("tot_time", 0.0))
        if "learning_rate" in runner_state:
            self.alg.learning_rate = runner_state["learning_rate"]
            for param_group in self.alg.optimizer.param_groups:
                param_group["lr"] = self.alg.learning_rate

        rng_state = loaded.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state["torch"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            if torch.cuda.is_available() and "cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        return loaded.get("infos")

    def log_video(self, it):
        if it - self.last_recording_it >= self.runner_cfg.save_video_interval:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            import numpy as np
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            print(video_array.shape)
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
            self.logger.log_video(video_array, step=it, fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert


# Backward-compatible public name used by existing scripts and checkpoints.
Runner = OnPolicyRunner

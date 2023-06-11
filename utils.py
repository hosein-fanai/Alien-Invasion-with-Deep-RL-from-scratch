from alien_invasion_env import AlienInvasionEnv

import tensorflow as tf

import gym
from gym.spaces import Box, Discrete
from gym.wrappers import FrameStack, GrayScaleObservation
from gym.wrappers import ResizeObservation

import numpy as np

# from collections import deque

import time


# class OnDiskReplayBuffer:
#     file_name_prefix = "buffer cache"
#     file_name_postfix = ".tfrecord"

#     def __init__(self, buffer_size):
#         self.queue = deque(maxlen=buffer_size)

#         self.buffer = []
#         self.cache_list = {}
#         self.steps_counter = 0

#     def append(self, replay):
#         self.buffer.append(replay)

#     def dump(self):
#         prev_step_count = self.steps_counter
#         steps_count = len(self.buffer)
#         self.steps_counter += steps_count

#         file_path = os.path.join(self.file_name_prefix, 
#                             f"{prev_step_count}_{self.steps_counter}")
#         file_path += self.file_name_postfix

#         for i in range(prev_step_count, self.steps_counter):
#             self.queue.append(i)
#             self.cache_list[i] = file_path

#         self._write_tfrecord_file(file_path)
#         self.buffer = []

#         self._clear_cache(prev_step_count, steps_count)

#     def get_random_replays(self, batch_size=64):
#         indices = np.random.randint(len(self.queue), size=batch_size)
        
#         return [self._read_tfrecord_file(self.queue[index]) for index in indices]

#     def _write_tfrecord_file(self, file_path):
#         for replay in self.buffer:
#             with tf.io.TFRecordWriter(file_path) as writer:
#                 feature_dict = {
#                     'consec_states': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(replay[0]).numpy()])),
#                     'action': tf.train.Feature(int64_list=tf.train.Int64List(value=[replay[1]])),
#                     'reward': tf.train.Feature(float_list=tf.train.FloatList(value=[replay[2]])),
#                     'done': tf.train.Feature(int64_list=tf.train.Int64List(value=[replay[3]])),
#                 }
                
#                 example = tf.train.Example(
#                     features=tf.train.Features(feature=feature_dict))
#                 serialized = example.SerializeToString()
#                 writer.write(serialized)

#     def _read_tfrecord_file(self, step_number):
#         file_path = self.cache_list[step_number]

#         feature_description = {
#             'consec_states': tf.io.FixedLenFeature([], tf.uint8),
#             'action': tf.io.FixedLenFeature([], tf.uint8),
#             'reward': tf.io.FixedLenFeature([], tf.float32),
#             'done': tf.io.FixedLenFeature([], tf.uint8),
#         }

#         def parse_example(serialized_example):
#             example = tf.io.parse_single_example(serialized_example, feature_description)
#             return example["consec_states"], example["action"], example["reward"], example["done"]

#         dataset = tf.data.TFRecordDataset(file_path)
#         dataset = dataset.map(parse_example, num_parallel_calls=10)
#         dataset = list(dataset)
        
#         return dataset[step_number]

#     def _clear_cache(self, start, size):
#         # for file_path in set(self.cache_list.values()):
#         #     file_name = os.path.split(file_path)[-1]
#         #     a, b = file_name.split('_')

#         #     if a < self.steps_counter and b < self.steps_counter:
#         #         os.remove(file_path)
#         pass


class AlienInvasionGymEnv(gym.Env):
    
    def __init__(self, **kwargs):
        self.game = AlienInvasionEnv(**kwargs)

        width, height = self.game.settings.screen_dims
        shape = ((height-50)//4, width//4, 1) if self.game.preprocess_obs else (width, height, 3)
        if self.game.render_type == "gray_scale":
            shape = (shape[0], shape[1], 1)

        self.observation_space = Box(low=0, high=255, shape=shape, dtype="uint8")
        self.action_space = Discrete(6)

    def step(self, action, mode="human"):
        return self.game.step(action, mode)

    def render(self, mode="human"):
        return self.game.render(mode)

    def reset(self, mode="human"):
        return self.game.reset(mode)

    def close(self):
        self.game.close()


class TransposeCropObservation(gym.ObservationWrapper):
    
    def __init__(self, env, crop_size=50, swap_time=True, swap_dims=True):
        super().__init__(env)
        self.crop_size = crop_size
        self.swap_time = swap_time
        self.swap_dims = swap_dims

        shape = env.observation_space.shape
        if self.swap_dims:
            if self.swap_time:
                shape = (shape[2]-self.crop_size, shape[1], shape[0])
            else:
                shape = (shape[0], shape[2]-self.crop_size, shape[1], shape[-1])
        else:
            if self.swap_time:
                shape = (shape[1], shape[2]-self.crop_size, shape[0])
            else:
                shape = (shape[0], shape[1], shape[2]-self.crop_size, shape[-1])

        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                            shape=shape, 
                                            dtype="uint8")

    def observation(self, observation):
        if self.swap_dims:
            if self.swap_time:
                observation = np.transpose(observation, (3, 2, 1, 0))[0, self.crop_size:, :]
            else:
                observation = np.transpose(observation, (0, 2, 1, 3))[:, self.crop_size:, :]
        else:
            if self.swap_time:
                observation = np.transpose(observation, (3, 1, 2, 0))[0, :, self.crop_size:]
            else:
                observation = np.transpose(observation, (0, 1, 2, 3))[:, :, self.crop_size:]

        return observation


class SkipFrame(gym.ObservationWrapper):
    
    def __init__(self, env, skip_frame=2):
        super().__init__(env)
        self.skip_frame = skip_frame

        shape = env.observation_space.shape
        shape = (*shape[:-1], shape[-1]//skip_frame)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                            shape=shape, 
                                            dtype="uint8")

    def observation(self, observation):
        if self.skip_frame > 1:
            # return observation[..., 1::self.skip_frame]
            return observation[..., ::-self.skip_frame][..., ::-1]
        else:
            return observation


class TruncateOnShiphit(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info["ship left"] < 3:
            done = True

        return obs, reward, done, info


# class ActionRepeat(gym.Wrapper):

#     def __init__(self, env, action_repeat):
#         super().__init__(env)
#         self.action_repeat = action_repeat

#     def step(self, action):
#         total_reward = 0
#         for _ in range(self.action_repeat):
#             obs, reward, done, info = self.env.step(action)
#             total_reward += reward
#             if done:
#                 break

#         return obs, total_reward, done, info


class ConvertTypeObservation(gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                            shape=env.observation_space.shape, 
                                            dtype="bfloat16")

    def observation(self, observation):
        return observation


def make_env(game_resolution=(1280, 720), preprocessing_type="env_defalut_preps", num_stack=4, reverse_time_dim=True, skipframe_div=1, truncate_on_shiphit=False):
    env = AlienInvasionGymEnv(
        game_resolution=game_resolution,
        preprocess_obs=True if preprocessing_type=="env_defalut_preps" else False
)

    if preprocessing_type == "gym_wrappers":
        height, width, _ = env.observation_space.shape
        env = ResizeObservation(env, shape=(height//4, width//4))
        env = GrayScaleObservation(env)

    # env = ActionRepeat(env, action_repeat=4)
    env = FrameStack(env, num_stack=num_stack)
    env = TransposeCropObservation(
        env, 
        crop_size=0 if preprocessing_type!="" else 50,
        swap_dims=False if preprocessing_type=="env_defalut_preps" else True,
        swap_time=reverse_time_dim
    )
    env = SkipFrame(env, skip_frame=skipframe_div)

    if truncate_on_shiphit:
        env = TruncateOnShiphit(env)

    return env


class DQNAgent:

    def __init__(self, env, model, target, optimizer, loss_fn, replay_buffer):
        self.env = env
        self.model =  model
        self.target = target
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.replay_buffer = replay_buffer

        self.action_space_num = self.env.action_space.n

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            Q_values = self.model(state[None])
            return tf.argmax(Q_values[0])

    # def _compress_two_consecutive_states(self, first_state, second_state, num_stack=4):
    #     state_shape = first_state.shape[:-1]
    #     consec_states = np.zeros((*state_shape, num_stack+1), dtype=np.uint8)

    #     consec_states[..., :-1] = first_state.copy()
    #     consec_states[..., -1] = second_state[..., -1].copy()

    #     del first_state, second_state

    #     return consec_states

    # def _decompress_two_consecutive_states(self, consec_states, num_stack=4):
    #     state_shape = consec_states.shape[:-1]
    #     first_state = np.zeros((*state_shape, num_stack), dtype=np.uint8)
    #     second_state = np.zeros((*state_shape, num_stack), dtype=np.uint8)

    #     first_state = consec_states[..., :-1]
    #     second_state = consec_states[..., 1:]

    #     return first_state, second_state

    def _play_one_step(self, state, epsilon=0.01, truncate_in_shiphit=False):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = self.env.step(action)

        if info["ship left"] < 3 and truncate_in_shiphit:
            done = True

        # consec_states = compress_two_consecutive_states(state, next_state)
        # replay_buffer.append((consec_states, action, reward, done))
        self.replay_buffer.append((state, action, reward, next_state, done))

        return next_state, reward, done, info

    def _play_multiple_steps(self, n_step, epsilon, truncate_in_shiphit):
        # obs = self.env.reset(mode="rgb_array")
        obs = self.env.reset()
        total_reward = 0

        frames = 0
        start = time.time()
        for step in range(n_step):
            obs, reward, done, _ = self._play_one_step(obs, epsilon, truncate_in_shiphit)
            total_reward += reward
            if done:
                break
            frames += 1
        fps = int(frames // (time.time() - start))

        # replay_buffer.dump()
        
        return total_reward, step, fps

    def _sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array(sub_exp) for sub_exp in zip(*batch)
        ]
        
        # batch = self.replay_buffer.get_random_replays(batch_size)
        # consec_states, actions, rewards, dones = [
        #     np.array(sub_exp) for sub_exp in zip(*batch)
        # ]
        # states, next_states = self.decompress_two_consecutive_states(consec_states)

        return states, actions, rewards, next_states, dones

    @tf.function
    def _train_step(self, batch_size, gamma):
        exp_batch = self._sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = exp_batch

        next_Q_values = self.model(next_states)
        next_best_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(next_best_actions, depth=self.action_space_num)

        next_best_Q_values = tf.reduce_sum(self.target(next_states) * next_mask, axis=1)

        target_Q_values = rewards + (1 - dones) * gamma * next_best_Q_values
        target_Q_values = tf.reshape(target_Q_values, (-1, 1))

        mask = tf.one_hot(actions, depth=self.action_space_num)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states, training=True)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def training_loop(self, iteration, n_step=10_000, batch_size=64, gamma=0.99, 
                    warmup=10, train_interval=1, target_update_interval=50, 
                    soft_update=False, truncate_in_shiphit=False, epsilon_fn=None):
        func = lambda episode: max(1-episode/(iteration*0.9), 0.01)
        epsilon_fn = func if epsilon_fn is None else epsilon_fn

        best_score = float("-inf")
        rewards = []
        all_loss = []

        for episode in range(warmup):
            _, step, fps = self._play_multiple_steps(n_step, 1.0, False)

            print(f"\r---Warmup---Episode: {episode}, Steps: {step}, FPS: {fps}", end="")

        for episode in range(iteration):
            epsilon = epsilon_fn(episode)

            total_reward, step, fps = self._play_multiple_steps(n_step, epsilon, truncate_in_shiphit)
            rewards.append(total_reward)

            if total_reward > best_score:
                self.model.save_weights(f"models/DQN_ep#{episode}_eps#{epsilon:.4f}_rw#{total_reward:.1f}.h5")
                best_score = total_reward

            if episode % train_interval == 0:
                loss = self._train_step(batch_size, gamma)
            all_loss.append(loss)

            if episode % target_update_interval == 0:
                if not soft_update:
                    self.target.set_weights(self.model.get_weights())
                else:
                    target_weights = self.target.get_weights()
                    online_weights = self.model.get_weights()
                    for index in range(len(target_weights)):
                        target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                    self.target.set_weights(target_weights)

            print(f"\rEpisode: {episode}, Steps: {step}, FPS: {fps}, Reward:{total_reward:.1f}, Epsilon: {epsilon:.4f}, Loss: {loss}", end="")

        return rewards, all_loss


if __name__ == "__main__":
    # env = AlienInvasionGymEnv()
    env = make_env()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    env.close()
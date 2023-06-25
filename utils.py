from alien_invasion_env import AlienInvasionEnv

import gym
from gym.spaces import Box, Discrete
from gym.wrappers import FrameStack, GrayScaleObservation
from gym.wrappers import ResizeObservation

import numpy as np


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
    
    def __init__(self, env, skip_frame=2, channels_last=True):
        super().__init__(env)
        self.skip_frame = skip_frame
        self.channels_last = channels_last

        shape = env.observation_space.shape
        if channels_last:
            shape = (*shape[:-1], shape[-1]//skip_frame)
        else:
            shape = (shape[0]//skip_frame, *shape[1:])
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                            shape=shape, 
                                            dtype="uint8")

    def observation(self, observation):
        if self.skip_frame > 1:
            if self.channels_last:
                # return observation[..., 1::self.skip_frame]
                return observation[..., ::-self.skip_frame][..., ::-1]
            else:
                return observation[::-self.skip_frame, ...][::-1, ...]
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
    env = SkipFrame(env, skip_frame=skipframe_div, channels_last=reverse_time_dim)

    if truncate_on_shiphit:
        env = TruncateOnShiphit(env)

    return env


if __name__ == "__main__":
    # env = AlienInvasionGymEnv()
    env = make_env()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    env.close()
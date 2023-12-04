### 基础 IsaacGym APIs

#### asset 加载与上色

```python
ant_asset = load_asset( ... )

self.ant_handles = []
self.envs = []

for i in range(self.num_envs):
    # create env instance
    env_ptr = self.gym.create_env(sim, lower, upper, num_per_row)
	ant_handle = gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 1, 0)

    for j in range(self.num_bodies):
        gym.set_rigid_body_color(env_ptr, ant_handle, j, 
                                 gymapi.MESH_VISUAL, 
                                 gymapi.Vec3(0.97, 0.38, 0.06))

self.envs.append(env_ptr)
self.ant_handles.append(ant_handle)
```



#### action

action 与 dof 关系非常紧密，通过 gym 获取 dof 的状态：

```python
actor_dof_state = gym.acquire_dof_state_tensor(self.sim)
self.dof_state = wrap_tensor(actor_dof_state).view(num_envs, num_dof, 2)
self.dof_pos = self.dof_state[..., 0]
self.dof_vel = self.dof_state[..., 1]
```


需要注意的是，action 一般需要经过归一化为 [0, 1]，其上下限获取一般是 dof 的上下限：

```python
dof_num = gym.get_asset_dof_count(ant_asset)

ant_handle = load_asset(...)

dof_limits_lower = []
dof_limits_upper = []
dof_prop = gym.get_actor_dof_properties(env_ptr, ant_handle)  # 其中一个
for j in range(dof_num):
    dof_limits_lower.append(min(dof_prop['upper'][j], dof_prop['lower'][j]))
	dof_limits_upper.append(max(dof_prop['upper'][j], dof_prop['lower'][j]))

self.dof_limits_lower = to_torch(dof_limits_lower, device=self.device)
self.dof_limits_upper = to_torch(dof_limits_upper, device=self.device)

# 在一些处理中，有时候会把 3-DOF rotation 上下限取为 [-pi, pi]
# 1-DOF rotation 上下限保持不变
```



应用 action：

- torque control：

  ```python
  # 获取 motor_efforts
  actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
  motor_efforts = [prop.motor_effort for prop in actuator_props]
  
  # motor_efforts 作为权重
  forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
  force_tensor = gymtorch.unwrap_tensor(forces)
  self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
  ```



- PD control：

  ```python
  # 计算 scale 和 offset
  self._pd_action_offset = 0.5 * (lim_high + lim_low)
  self._pd_action_scale = 0.5 * (lim_high - lim_low)
  
  # 应用 PD control 的时候缩放
  pd_tar = self._pd_action_offset + self._pd_action_scale * action
  pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
  gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
  ```



#### observation

observation 可以有多种取值的类型，一般而言包括两部分

- avatar 本身的属性，如 root state, non-root (other joints) state, force, ...
- task 本身相关的属性，如 distance to target，root yaw pitch roll, ...



##### avatar 属性获取

```python
actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
actor_dof_state = self.gym.acquire_dof_state_tensor(self.sim)
actor_force_sensor = self.gym.acquire_force_sensor_tensor(self.sim)
# 这里使用的是 ant 模型，有四条腿四个力传感器，humanoid 有两个
force_sensor = wrap_tensor(actor_force_sensor).view(self.num_envs, 4 * 6)

# wrap_tensors ...
```

计算 observation：

```python
root_states,         # root (position 3, rotation 4, linear vel 3, angular vel 3)
dof_pos,             # joint position (D)
dof_vel,             # joint velocity (D)
inv_start_rot,       # inv(start root rotation)  (4)
					 # 针对根节点，有时候需要计算根节点的相对（第一帧）运动而不是当前状态
force_sensor,        # feet force sensor (force 3, torque 3)

dof_limits_lower,    # dof lower bound
dof_limits_upper,    # dof upper bound

dof_vel_scale,       # #
contact_force_scale, # #

actions,             # -
```



##### task 属性  (Ant 为例)

```python
targets,               # + task target
potentials,            # + potentials
dt,                    # -
basis_vec0,            # + heading
basis_vec1,            # + up vec
up_axis_idx            # -
```



### 创建自定义任务

所有 Task 继承自 `VecTask`，并需要重载以下三个函数：`create_sim`，`pre_physics_step`和`post_physics_step`

```python
class MyNewTask(VecTask):
	def __init__(self, config_dict):
		super().__init__(cfg=config_dict)

    def create_sim(self):
        # 设置 up-axis
        # 调用 super().create_sim
        # 创建地板
        # 加载资源

    def pre_physics_step(self, actions):
        # 应用 action

    def post_physics_step(self):
        
        # progress buf 和 randomize buf
		self.progress_buf += 1
        self.randomize_buf += 1
        
        # 检查是否有需要 reset 的
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0: self.reset_idx(env_ids)
        
        # 计算&存储 observation (S_{t+1}), reward (R_t) 和 reset (Done_t)
        self.obs_buf[:] = get_observation(...)
        self.rew_buf[:] = get_reward(...)
        self.reset_buf[:] = get_reset(...)
        
        # 其他 buffer：
        # self.states_buf : Tensor
        # self.timeout_buf : Tensor
        # self.extras : dict
        
        # ----
        # debug 可视化的渲染
        
	# --------- 其他函数 --------- #
    def reset_idx(self, env_idx):
		# 主要是设置 dof_state 和 root_state
		
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
		# root_state:
		self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )

        # dof_state：
        self.dof_pos[env_ids] = get_init_pos()
        self.dof_vel[env_ids] = get_init_vel()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        
        # 其他：
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # 设置其他自定义的状态
        # ...

```



并且，需要将相应的 task 加入 `isaacgym_task_map`：

```python
from isaacgymenvs.tasks.my_new_task import MyNewTask
...
isaac_gym_task_map = {
    'Anymal': Anymal,
    # ...
    'MyNewTask': MyNewTask,
}
```

注意：以上的方法并不能成功添加 task，需要增加相应的配置文件：

- Task:  `MyNewTask.yaml`

  ```yaml
  name: MyNewTask  # 在这里修改名称
  # ...
  
  env:
      # num_envs, epicoseLength, dofVelScale, deathCost, ...
      # 可以加入自定义的参数，比如 reward function scale 等等
  
  sim:
  	# 把 gpu 换成 cpu 可以在窗口查看 dof 的详情
  	use_gpu_pipeline: ${eq:${...pipeline},"gpu"}
  	
      # up_axis, ...
  
  task:
  	# domain randomization, ...
  ```

- rl-games: `MyNewTaskPPO.yaml`

  ```yaml
  params:
  	seed: ${...seed}
  
  	algo:
  		name: a2c_continuous  # 算法名称，需要注册
  	
  	model:
  		name: continous_a2c_logstd  # 模型名称，需要注册
  		
  	network:
  		name: actor_critic	# 网络名称，需要注册
  		separate: False
  		
  		space:
              continuous:
                  mu_activation: None
                  sigma_activation: None
  
              mu_init:
  	            name: default
              sigma_init:
  	            name: const_initializer
  				val: 0
  			fixed_sigma: True
  		mlp:  # 可以替换
              units: [256, 128, 64]
              activation: elu
              d2rl: False
  
  			initializer:
                  name: default
  			regularizer:
  				name: None
  
      load_checkpoint: ...
      load_path: ...
  
      config:
  	    name: ${resolve_default:MyNewTask,${....experiment}}  # 修改名称
  	    ppo: True
      	num_actors: ${....task.env.numEnvs}
          gamma: 0.99
          tau: 0.95
  	    learning_rate: 3e-4
  	    # ...
  ```



启动训练：

`python ./isaacgymenvs/train.py task=RMAnt num_envs=4096 +minibatch_size=65536 `

其中，不带 `+` 的参数覆盖 RMAnt.yaml 的参数设置 `+` 表示增加新的参数，覆盖 rl-games 的 RMAntPPO.yaml 

测试：

`python train.py task=Ant num_envs=64 checkpoint=runs/Ant/nn/Ant.pth`

其他：

`sim_device`:  "cuda:0"

`rl_device`: "cuda:0"

可以选取其他值，在其他 GPU 上训练



#### 自定义算法和模型

- AlgoObserver
- Agent
- ModelBuilder
- NetworkBuilder
- Player

```c++
torch_runner {
    /* ---------------- 训练时 ---------------- */
	run_train {
        agent = algo_factory.create();  // agent 工厂
        _restore();          // 加载 ckpt
        _override_sigma();   // 将 fixed_sigma 搬运到 sigma，硬编码
        agent.train();       // train 入口
	}

    /* ---------------- 测试时 ---------------- */
	run_play {
        player = player_factory.create();  // player 工厂
        _restore();         // 加载 ckpt
        _override_sigma();  // 将 fixed_sigma 搬运到 sigma，硬编码
        player.run();       // play 入口        
	}
}
```



```c++
algo_builder.Agent.__init__()  # CommonAgent
{
	self.model = model_builder:model.Model.build()  /* ModelAMPContinuous */
    {
    	// 负责收集网络的输出，放置在 replay buffer 中
        return ModelBuilder.Network(model_builder:network.Builder.build()/* AMPBuilder */)
        {
            // 只会构建并生成相应的网络
            return NetworkBuilder.Network(
                - actor_mlp
                - critic_mlp
                - disc_mlp
                - sigma
                - value.RunningMeanStd
                - obs.RunningMeanStd
                - ...
            )
        }
        
        self.dataset = AMPDataset(...)  // MARK: dataset
    }
}

```

```
dataset: 训练 actor
experience_buffer：训练 critic, discriminator
```



```c++
algo_builder.Agent.train()
{
    init_tensors();  // 初始化 experience buffer (!= replay buffer)
    while (true) {
	    train_epoch() {
            set_eval();
            for i in range(horizon_length):
			   // experience_buffer.update ... 
            	- env_reset_done()
            	- action = model.forward( obs ) {
                    return NetworkBuilder.Network.forward( obs )
                }
            	- env_step() {
                    MyTask.post_physics_step()
                }
            	- v_{t+1} = eval_critic( obs )
            	- 
            	- 
            	- 
            	- 
                // no need to train like a real GAN
            	- Loss = actor_loss
                       + critic_loss
                       - entropy_loss
                       + bound_loss  // clip action mu
                       + disc_loss   // class_loss + GP_loss
        }
    }
}
```






| Key           | Shape  | Value       | Desc  |
| ------------- | ----------------------------------- | ------------ | ------------- |
| actions       |                                     |              |              |
| neglogpacs    |                                     |              |              |
| values        |                                     |              |              |
| returns       |                                     |              |              |
| rewards       |                                     |              |              |
| obses         | [H, E, 105]                         | s_t          |           |
| next_obses    | [H, E, 105]                         | s_t+1        |         |
| amp_obs       | [H, E, 210]                         | (s_t+1, s_t) | amp : Discriminator |
| amp_obs_replay | [HxE, 210] |  | 从 replay buffer 采样 |
| amp_obs_demo | [HxE, 210] |              | 真实数据 demo |
| dones         | [H, E]                              |              |              |
| mus           |                                     |              |              |
| sigmas        |                                     |              |              |
| played_frames |                                     |              |              |
| disc_rewards  |                                     |              |              |
|               |                                     |              |              |





### Motion

*.npy file

```
rotation:                [F, J, 4], float32
root_translation:        [F, 3], float32
global_velocity:         [F, J, 3], float32
global_angular_velocity: [F, J, 3], float32
skeleton_tree:
	node_names:          string[J]  # e.g. [pelvis, torso, head, right_upper_arm, ...]
	parent_indices:      [J], int64
	local_translation:   [J, 3], float32
fps: <ndarray.int32>  # 60
```

```python
motion_ids = _motion_lib.sample_motion(num_envs)
# 训练可能有多个 motion，每个 motion 有各自的 _motion_weights
# 这里按照 _motion_weights 概率采样

motion_times = _motion_lib.sample_time(motion_ids)
# 输入：motion 片段的 id
# 输出：该片段采样的时间点 (random state init)
#      note：可以置为 0 (fixed state init)
# 
# 随机采样一个时间点

root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = _motion_lib.get_motion_state(motion_ids, motion_times)
# 获取当前 id 当前 time 的 motion state
```




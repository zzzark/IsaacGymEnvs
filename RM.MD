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

`python ./isaacgymenvs/train.py task=RMAnt`

测试：

`python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth`



#### 自定义算法和模型


algo_observer: 算法监视器，用于纪录日志等

algo (Agent)

player (Player)

model (Model)

network (Builder)

```python
runner = Runner(algo_observer)
runner.algo_factory.register_builder(
    'amp_continuous', 
    lambda **kwargs : amp_continuous.AMPAgent(**kwargs)
)
runner.player_factory.register_builder(
    'amp_continuous',
    lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs)
)
model_builder.register_model(
    'continuous_amp', 
    lambda network, **kwargs : amp_models.ModelAMPContinuous(network)
)
model_builder.register_network(
    'amp', 
    lambda **kwargs : amp_network_builder.AMPBuilder()
)

```






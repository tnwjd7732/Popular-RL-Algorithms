import env as environment
import parameters as params
import my_ppo
import my_dqn

env = environment.Env()
params.cloud = 1
ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

print("start loading")
dqn = dqn_.load_model(params.dqn_path)
ppo = ppo_.load_model(params.ppo_path)

dqn_.save_model(params.staticClst_dqn_path)
ppo_.save_model(params.staticClst_ppo_path)
print("end of saving model")
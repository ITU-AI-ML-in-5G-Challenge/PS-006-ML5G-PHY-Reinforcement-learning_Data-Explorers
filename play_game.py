import numpy as np

def play_game(env, TrainNet, TargetNet, epsilon, copy_step, k):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    for i in range(k):
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        print("action: ",action)
        print("current episode: ", env.caviar_bs.UEs[0].episodeID)
        print("current step: ", i)
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if i != k-1:
            done = False
        else:
            done =  True
        print("Done: ", done)
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, np.mean(losses)
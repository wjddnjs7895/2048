import os, time ,threading, random, math
import tensorflow as tf
import Client_2048 as client
import numpy as np
from collections import deque

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Conv2D, Flatten

from tensorflow.python.client import device_lib

class DQN(tf.keras.Model) :
    def __init__(self,action_size) : 
        super(DQN, self).__init__()
        self.conv1 = Conv2D(128, (2,2), strides = (1,1), activation = 'relu', input_shape = (4,4,1))
        self.conv2 = Conv2D(128, (2,2), strides = (1,1), activation = 'relu')
        self.flatten = Flatten()
        self.fc = Dense(256, activation = 'relu')
        self.fc_out = Dense(action_size, kernel_initializer = RandomUniform(-1e-3, 1e-3))

    def call(self,x) : 
        x = tf.expand_dims(x, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q

class DQNAgent : 
    def __init__(self, state_size, action_size) : 
        self.render = False

        self.state_size = state_size
        self.action_size = action_size
        
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 1000000
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.train_start = 50000
        self.update_target_rate = 10000

        self.memory = deque(maxlen = 100000)

        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(lr = self.learning_rate, clipnorm = 10.)

        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0    
        self.writer = tf.summary.create_file_writer('summary/game_2048_dqn')
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    def update_target_model(self) : 
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state) : 
        if np.random.rand() <= self.epsilon : 
            return random.randrange(self.action_size)
        else : 
            arr = np.array(state,dtype = np.float32)

            arr = tf.expand_dims(arr, 0)
            q_value = self.model(arr)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done) : 
        self.memory.append((state, action, reward, next_state, done))

    def draw_tensorboard(self, score, step, episode) : 
        with self.writer.as_default() : 
            tf.summary.scalar('Total Reward/Episode', score, step = episode)
            tf.summary.scalar('Average Max Q/Episode', self.avg_q_max / float(step), step = episode)
            tf.summary.scalar('Duration/Episode', step, step = episode)
            tf.summary.scalar('Average Loss/Episode', self.avg_loss / float(step), step = episode)

    def train_model(self) : 
        if self.epsilon > self.epsilon_end : 
            self.epsilon -= self.epsilon_decay_step
        
        mini_batch = random.sample(self.memory, batch_size)

        states = np.array([sample[0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape : 
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis = 1)

            target_predicts = self.target_model(next_states)

            max_q = np.amax(target_predicts, axis = -1)
            targets = rewards + (1-dones) * self.discount_factor * max_q

            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            self.avg_loss += loss.numpy()
        

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

if __name__ == "__main__" : 
    tf.keras.backend.set_floatx('float32')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.compat.v1.Session(config = config)
    session.close()

    state_size = client.WIDTH * client.HEIGHT
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    global_step = 0
    score_avg = 0
    score_max = 0
    num_episode = 200000
    batch_size = 100
     
    for e in range(num_episode) : 
        score = 0 
        step = 0
        client.BOARD = np.zeros((client.WIDTH, client.HEIGHT))
        state = client.BOARD
        client.game_start()
        done = False
        print("\nEpisode number : #",e)

        while not done :

            global_step += 1
            step += 1
            temp_board = client.BOARD.copy()
            for i in range(4) : 
                for j in range(4) : 
                    if temp_board[i,j] : 
                        temp_board[i,j] = np.log2(temp_board[i,j])
            action = agent.get_action(temp_board)
            state, reward, next_state, done = client.game_main_loop(action)
            if reward : 
                score += reward

            for i in range(4) : 
                for j in range(4) : 
                    if state[i,j] : 
                        state[i,j] = np.log2(state[i,j])
                    if next_state[i,j] : 
                        next_state[i,j] = np.log2(next_state[i,j])

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start : 
                agent.train_model()

                if global_step % agent.update_target_rate == 0 : 
                    agent.update_target_model()

            if done : 
                if global_step > agent.train_start : 
                    agent.draw_tensorboard(score, step, e)

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg!= 0 else score
                score_max = score if score > score_max else score_max

                log = "episode : {:5d} | ".format(e)
                log += "score : {:4.1f} | ".format(score)
                log += "score max : {:4.1f} | ".format(score_max)
                log += "score avg : {:4.1f} | ".format(score_avg)
                log += "memory length : {:5d} | ".format(len(agent.memory))
                log += "epsilon : {:.3f} | ".format(agent.epsilon)
                log += "avg loss : {:3.2f} | ".format(agent.avg_loss / float(step))
                print(log) 

                agent.avg_q_max, agent.avg_loss = 0, 0


        if e % 1000 == 0 :         
            agent.model.save_weights("./save_model/model",save_format = "tf")
        if e % 10 == 0 : 
            print('\n------------------------------------------------------------------------------------------------------------------------------------')
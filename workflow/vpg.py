import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import gym
import tensorflow as tf
from ..funcs.variable import *


import kmc_env
import kmcsim
from kmcsim.sim import KMCModel
from kmcsim.sim import EventTree
from kmcsim.sim import RunSim
from kmc_env.envs.kmcsim_state_funcs import make_surface_proj,calc_roughness,get_state_reward,get_incremented_rates,gaussian
from kmc_env.envs.kmc_env import *

class pg:

    def __init__(self,box = [16, 32, 4],box_extension=32,target_roughness=0.98,episodes=150,wdir=r"C:\Users\ni1\Documents\RL\kmcsim\data\working",
        reward_type='gaussian',reward_multiplier=1000,reward_tolerance=2,rates_spread=0.1,rates_adjustment=1):
        self.box=box
        self.box_extension=box_extension
        self.target_roughness=target_roughness
        self.episodes=episodes
        self.wdir=wdir
        self.env = KmcEnv(box=self.box,box_extension=self.box_extension,target_roughness=self.target_roughness,
             reward_type=reward_type,reward_multiplier=reward_multiplier,reward_tolerance=reward_tolerance,
             rates_spread=rates_spread,rates_adjustment=rates_adjustment,folder_with_params=self.wdir)

        self.state,self.reward = self.env.reset()
        self.state_size = self.env.state.shape
        self.dim=self.env.state.shape[0]*self.env.state.shape[1]
        self.dim_actions=27
        self.learning = np.zeros((self.episodes,1))

    def set_parameters(self,num_nodes = 100,maxsteps=10,num_gradients = 1,num_runs = 150):
        self.num_nodes=num_nodes
        self.maxsteps=maxsteps
        self.num_gradients=num_gradients

    def start_session(self):

        self.sess = tf.InteractiveSession()
        self.state_vpg = tf.placeholder(tf.float32, shape=[None, self.dim])
        self.action_choice = tf.placeholder(tf.float32, shape=[None, self.dim_actions])
        self.reward_signal = tf.placeholder(tf.float32, shape=(None,1) )
        self.n_timesteps = tf.placeholder(tf.float32, shape=())

        self.W1 = weight_variable([self.dim, self.num_nodes])
        self.b1 = bias_variable([self.num_nodes])
        self.a1 = tf.nn.relu(tf.matmul(self.state_vpg, self.W1) + self.b1)

        self.W0 = weight_variable([self.num_nodes, self.dim_actions])
        self.b0 = bias_variable([self.dim_actions])
        self.a0 = tf.nn.softmax(tf.matmul(self.a1, self.W0) + self.b0)

        log_prob = tf.log(tf.diag_part(tf.matmul(self.a0, tf.transpose(self.action_choice))))
        log_prob = tf.reshape(log_prob, (1,-1))
        loss = tf.matmul(log_prob, self.reward_signal)
        loss = -tf.reshape(loss, [-1])

        self.loss=loss

        self.train_step = tf.train.AdamOptimizer().minimize(loss)
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
    
    def run_session_once(self):
        self.states = np.zeros((self.maxsteps,self.dim), dtype='float32')
        self.actions = np.zeros((self.maxsteps,self.dim_actions), dtype='float32')
        self.rewards = np.zeros((self.maxsteps,1), dtype='float32')

        self.timestep =0
        self.observation,_ = self.env.reset()
        self.observation = np.reshape(self.observation,(1,self.dim))
        self.observation=self.observation-self.observation.mean()
        done = False
        
        while not done:
   
            self.action_prob = self.sess.run(self.a0, feed_dict={self.state_vpg: self.observation})     
            self.action_prob-=self.action_prob.min()
            self.action_prob/=np.sum(self.action_prob)
            action = np.argmax(np.random.multinomial(1, self.action_prob[0]-0.0001))
            self.actions[self.timestep, action] = 1

            x1=int(action%3)
            action=(action-x1)//3
            x2=int(action%3)
            action=(action-x2)//3
            x3=int(action%3)
            
            action=[x1,x2,x3]
            
            self.new_observation, self.reward, done = self.env.step(action)
            self.new_observation=self.new_observation-self.new_observation.mean()
            self.states[self.timestep, :] = self.observation
        
            
            self.rewards[self.timestep, :] = self.reward
            self.timestep += 1
            self.new_observation = np.reshape(self.new_observation,(1,self.dim))
            self.observation[:] = self.new_observation
            print('Timestep: {}, reward: {:.4}, RMS: {:.3}, action: {}'.format(self.timestep,self.reward,calc_roughness(self.observation),action))

        self.states = self.states[:self.timestep, :]
        self.actions = self.actions[:self.timestep, :]
        self.rewards = self.rewards[:self.timestep,:]
        self.rewards[:, 0] = np.cumsum(self.rewards[::-1])[::-1]

        self.gradients={"W1":[],"b1":[],"a1":[],"W0":[],"b0":[],"a0":[]}

        for i in range(self.num_gradients):
            self.sess.run(self.train_step, feed_dict={self.state_vpg: self.states, self.action_choice: self.actions, self.reward_signal: self.rewards, self.n_timesteps: self.timestep})



            for n,var in enumerate([self.W1,self.b1,self.a1,self.W0,self.b0,self.a0]):
                var_grad = tf.gradients(self.loss, [var])[0]
                self.var_grad_val = self.sess.run(var_grad, feed_dict={self.state_vpg: self.states, self.action_choice: self.actions, self.reward_signal: self.rewards, self.n_timesteps: self.timestep})       
                self.gradients[["W1","b1","a1","W0","b0","a0"][n]]+=[self.var_grad_val]
            print('VPG is running!')            
       

    def run_session(self):
        for run in range(self.episodes):
            self.run_session_once()
            self.learning[run] = self.timestep

    def save_session(self,name='sess'):
        saver = tf.train.Saver()
        saver.save(self.sess, name)

    def load_session(self,name='sess'):
        self.sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(name)
        saver.restore(self.sess,tf.train.latest_checkpoint('./'))

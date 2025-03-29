% mainSimulation.m
clc;
clear;

% Add helper functions to path (if saved in separate files)
addpath(genpath('functions'));

% Network Parameters
num_nodes = 50;
max_iterations = 100; % Number of learning iterations
alpha = 0.1; % Learning rate
gamma = 0.9; % Discount factor
epsilon = 0.2; % Exploration rate
k = 1; % For softmax exploration

% Initialize States, Actions, Q-Table
num_states = num_nodes; % Assume each node is a state
num_actions = num_nodes; % Actions represent possible cluster head choices
Q = zeros(num_states, num_actions); % Q-table

% Reward Function Parameters
energy_levels = rand(1, num_nodes) * 100; % Random initial energy levels
threshold = 20; % Threshold for low energy

% Simulation Loop
for iteration = 1:max_iterations
    for current_node = 1:num_nodes
        % Step 1: Choose Action (Cluster Head Selection)
        action = epsilon_greedy(Q, current_node, epsilon);

        % Step 2: Execute Action
        [reward, next_node, energy_levels] = execute_action(current_node, action, energy_levels, threshold);

        % Step 3: Update Q-Table
        Q = update_Q(Q, current_node, action, reward, next_node, alpha, gamma);
    end
end

% Final Cluster Heads
[~, cluster_heads] = max(Q, [], 2);
disp('Cluster Heads:');
disp(unique(cluster_heads));

% Routing Example (Node 30 to Node 50)
start_node = 30;
end_node = 50;
max_steps = 100; % Maximum steps to avoid infinite loops
route = rl_routing(Q, start_node, end_node, max_steps);
disp('Best Route:');
disp(route);

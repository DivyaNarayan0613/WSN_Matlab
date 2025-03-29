clc; clear; close all;

% Initialize Network Parameters
num_nodes = 100;
network_area = [100, 100];
max_iterations = 50;
energy_threshold = 20;
num_clusters = 10;
min_alive_nodes = 20; % Stop if fewer than 20 nodes are alive

% Random Deployment of Nodes
nodes = rand(num_nodes, 2) .* repmat(network_area, num_nodes, 1);
energy_levels = randi([50, 100], num_nodes, 1);
initial_energy_levels = energy_levels;

% Reinforcement Learning Parameters
Q = zeros(num_nodes, num_nodes);
gamma = 0.9;
alpha = 0.1;
epsilon = 0.1;

% Cluster Head Selection
cluster_heads = randperm(num_nodes, num_clusters); % Initial CHs (row vector)

% Assign nodes to nearest CH
distances = pdist2(nodes, nodes(cluster_heads, :));
[~, cluster_assignments] = min(distances, [], 2);

% Reinforcement Learning Training
fitness_values = zeros(1, max_iterations);
alive_nodes = num_nodes;

for iter = 1:max_iterations
    % Stop if too many nodes are dead
    alive_nodes = sum(energy_levels > 0);
    if alive_nodes < min_alive_nodes
        disp(['Stopping at iteration ', num2str(iter), ': Too few alive nodes (', num2str(alive_nodes), ')']);
        max_iterations = iter; % Adjust for plotting
        fitness_values = fitness_values(1:iter);
        break;
    end

    for i = 1:num_nodes
        % Skip if node is dead
        if energy_levels(i) <= 0
            continue;
        end

        % CH Replacement
        if ismember(i, cluster_heads) && energy_levels(i) < energy_threshold
            ch_idx = find(cluster_heads == i, 1);
            cluster_nodes = find(cluster_assignments == ch_idx);
            viable_nodes = cluster_nodes(energy_levels(cluster_nodes) > energy_threshold);
            if ~isempty(viable_nodes)
                [~, max_energy_idx] = max(energy_levels(viable_nodes));
                new_ch = viable_nodes(max_energy_idx);
                cluster_heads(ch_idx) = new_ch; % Substitute CH
                % Reassign nodes to new CH
                distances = pdist2(nodes, nodes(cluster_heads, :));
                [~, cluster_assignments] = min(distances, [], 2);
                disp(['Iteration ', num2str(iter), ': Replaced CH ', num2str(i), ' with ', num2str(new_ch)]);
            else
                disp(['Iteration ', num2str(iter), ': No viable CH replacement for ', num2str(i)]);
            end
        end

        % Q-Learning: Choose action
        if rand < epsilon
            action = randi(num_nodes);
        else
            [~, action] = max(Q(i, :));
        end

        % Reward: Incorporate energy and distance
        if energy_levels(action) > 0
            dist = norm(nodes(i, :) - nodes(action, :));
            reward = 1 / (1 + dist) * (energy_levels(action) / 100);
        else
            reward = -1; % Penalty for dead node
        end

        % Q-Table Update
        Q(i, action) = Q(i, action) + alpha * (reward + gamma * max(Q(action, :)) - Q(i, action));

        % Update Energy Consumption
        if ismember(i, cluster_heads)
            energy_levels(i) = max(energy_levels(i) - randi([3, 7]), 0);
        else
            energy_levels(i) = max(energy_levels(i) - randi([1, 3]), 0);
        end
    end

    % Fitness: Combine Q-values, energy, and coverage
    coverage = mean(min(pdist2(nodes, nodes(cluster_heads, :)), [], 2)); % Avg distance to nearest CH
    fitness_values(iter) = sum(Q, 'all') + mean(energy_levels) - coverage;
end

% Optimized Routing Path: Use Q-table to find path
start_node = randi(num_nodes);
end_node = randi(num_nodes);
route = [start_node];
current = start_node;
for step = 1:num_nodes
    if current == end_node
        break;
    end
    [~, next] = max(Q(current, :));
    if energy_levels(next) <= 0 || ismember(next, route)
        break; % Avoid dead nodes or loops
    end
    route = [route, next];
    current = next;
end

%% Plots and Visualizations
% 1. Fitness Value Over Iterations
figure;
plot(1:max_iterations, fitness_values, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
grid on;
xlabel('Iterations');
ylabel('Fitness (Q-Values + Energy - Coverage)');
title('Fitness Convergence Over Iterations');
legend('Fitness Value');

% 2. Energy Levels Before & After
figure;
bar([initial_energy_levels, energy_levels]);
grid on;
xlabel('Node Index');
ylabel('Energy Level');
title('Energy Levels Before and After Optimization');
legend('Before Optimization', 'After Optimization');

% 3. Cluster Head Selection Visualization
figure;
scatter(nodes(:, 1), nodes(:, 2), 50, 'bo', 'filled'); hold on;
scatter(nodes(cluster_heads, 1), nodes(cluster_heads, 2), 100, 'ro', 'filled');
grid on;
xlabel('X Coordinate (meters)');
ylabel('Y Coordinate (meters)');
title('Final Cluster Head Selection');
legend('Nodes', 'Cluster Heads');
hold off;

% 4. Optimized Routing Path
figure;
plot(nodes(route, 1), nodes(route, 2), '-s', 'LineWidth', 2, 'MarkerSize', 10);
grid on;
xlabel('X Coordinate (meters)');
ylabel('Y Coordinate (meters)');
title('Optimized Routing Path');
legend('Routing Path');
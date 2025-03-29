clc; clear; close all;

% Initialize Network Parameters
num_nodes = 100;
network_area = [100, 100];
max_iterations = 50;
energy_threshold = 20;
num_clusters = 10;
min_alive_nodes = 20;
base_station = [50, 50]; % Base station at center
transmission_range = 30; % Max transmission range (meters)

% Random Deployment of Nodes
nodes = rand(num_nodes, 2) .* repmat(network_area, num_nodes, 1);
energy_levels = randi([50, 100], num_nodes, 1);
initial_energy_levels = energy_levels;

% Sensor Data (e.g., temperature readings)
sensor_data = randi([20, 30], num_nodes, 1); % Random data between 20-30

% Reinforcement Learning Parameters
Q = zeros(num_nodes, num_nodes);
gamma = 0.9;
alpha = 0.1;
epsilon = 0.1;

% Energy Model Parameters
E_elec = 50e-9; % Energy per bit for electronics (J/bit)
E_amp = 100e-12; % Energy for amplifier (J/bit/m^2)
data_size = 4000; % Data packet size in bits (e.g., 500 bytes)
agg_cost = 5e-9; % Energy cost per bit for aggregation (J/bit)
comp_cost = 2e-9; % Energy cost per bit for compression (J/bit)

% Cluster Head Selection
cluster_heads = randperm(num_nodes, num_clusters); % Initial CHs
distances = pdist2(nodes, nodes(cluster_heads, :));
[~, cluster_assignments] = min(distances, [], 2);

% Reinforcement Learning Training
fitness_values = zeros(1, max_iterations);
alive_nodes = num_nodes;
data_at_base = zeros(max_iterations, num_clusters); % Decompressed data at base station
compression_ratios = zeros(max_iterations, num_clusters); % Track compression ratio
ref_values = zeros(1, num_clusters); % Reference values for differential compression

for iter = 1:max_iterations
    % Stop if too many nodes are dead
    alive_nodes = sum(energy_levels > 0);
    if alive_nodes < min_alive_nodes
        disp(['Stopping at iteration ', num2str(iter), ': Too few alive nodes (', num2str(alive_nodes), ')']);
        max_iterations = iter;
        fitness_values = fitness_values(1:iter);
        data_at_base = data_at_base(1:iter, :);
        compression_ratios = compression_ratios(1:iter, :);
        break;
    end

    % Data Transmission and Aggregation
    aggregated_data = zeros(1, num_clusters); % Aggregated data per cluster
    for i = 1:num_nodes
        % Skip if node is dead
        if energy_levels(i) <= 0
            continue;
        end

        % Find assigned CH
        ch_idx = cluster_assignments(i);
        ch_node = cluster_heads(ch_idx);

        % Skip if CH is dead
        if energy_levels(ch_node) <= 0
            continue;
        end

        % Transmit data to CH
        dist_to_ch = norm(nodes(i, :) - nodes(ch_node, :));
        if dist_to_ch <= transmission_range
            % Energy to transmit (node -> CH)
            energy_tx = E_elec * data_size + E_amp * data_size * dist_to_ch^2;
            energy_levels(i) = max(energy_levels(i) - energy_tx, 0);

            % Energy for CH to receive
            energy_rx = E_elec * data_size;
            energy_levels(ch_node) = max(energy_levels(ch_node) - energy_rx, 0);

            % Aggregate data at CH (e.g., sum)
            aggregated_data(ch_idx) = aggregated_data(ch_idx) + sensor_data(i);
        end
    end

    % CHs Compress and Transmit to Base Station
    for c = 1:num_clusters
        ch_node = cluster_heads(c);
        if energy_levels(ch_node) <= 0
            continue;
        end

        % Aggregate energy cost
        energy_agg = agg_cost * data_size;
        energy_levels(ch_node) = max(energy_levels(ch_node) - energy_agg, 0);

        % Differential Compression
        if iter == 1 || ref_values(c) == 0
            % First iteration: Send full value as reference
            compressed_data = aggregated_data(c);
            ref_values(c) = aggregated_data(c);
            data_size_compressed = data_size; % No compression for reference
        else
            % Subsequent iterations: Send delta
            delta = aggregated_data(c) - ref_values(c);
            % Assume delta can be encoded with fewer bits (e.g., 50% of original size)
            data_size_compressed = data_size * 0.5; % Simplified compression ratio
            compressed_data = delta;
            % Update reference for next iteration
            ref_values(c) = aggregated_data(c);
        end

        % Compression energy cost
        energy_comp = comp_cost * data_size;
        energy_levels(ch_node) = max(energy_levels(ch_node) - energy_comp, 0);

        % Compute compression ratio
        compression_ratios(iter, c) = data_size_compressed / data_size;

        % Transmit to base station
        dist_to_base = norm(nodes(ch_node, :) - base_station);
        if dist_to_base <= transmission_range
            % Energy to transmit (CH -> base station)
            energy_tx = E_elec * data_size_compressed + E_amp * data_size_compressed * dist_to_base^2;
            energy_levels(ch_node) = max(energy_levels(ch_node) - energy_tx, 0);

            % Decompress at base station
            if iter == 1 || data_at_base(max(1, iter-1), c) == 0
                data_at_base(iter, c) = compressed_data; % Reference value
            else
                data_at_base(iter, c) = data_at_base(iter-1, c) + compressed_data; % Add delta
            end
        end
    end

    % CH Replacement
    for i = 1:num_nodes
        if ismember(i, cluster_heads) && energy_levels(i) < energy_threshold
            ch_idx = find(cluster_heads == i, 1);
            cluster_nodes = find(cluster_assignments == ch_idx);
            viable_nodes = cluster_nodes(energy_levels(cluster_nodes) > energy_threshold);
            if ~isempty(viable_nodes)
                [~, max_energy_idx] = max(energy_levels(viable_nodes));
                new_ch = viable_nodes(max_energy_idx);
                cluster_heads(ch_idx) = new_ch;
                distances = pdist2(nodes, nodes(cluster_heads, :));
                [~, cluster_assignments] = min(distances, [], 2);
                disp(['Iteration ', num2str(iter), ': Replaced CH ', num2str(i), ' with ', num2str(new_ch)]);
            else
                disp(['Iteration ', num2str(iter), ': No viable CH replacement for ', num2str(i)]);
            end
        end
    end

    % Q-Learning
    for i = 1:num_nodes
        if energy_levels(i) <= 0
            continue;
        end
        if rand < epsilon
            action = randi(num_nodes);
        else
            [~, action] = max(Q(i, :));
        end
        if energy_levels(action) > 0
            dist = norm(nodes(i, :) - nodes(action, :));
            reward = 1 / (1 + dist) * (energy_levels(action) / 100);
        else
            reward = -1;
        end
        Q(i, action) = Q(i, action) + alpha * (reward + gamma * max(Q(action, :)) - Q(i, action));
    end

    % Fitness: Include data transmitted and compression efficiency
    coverage = mean(min(pdist2(nodes, nodes(cluster_heads, :)), [], 2));
    avg_compression = mean(compression_ratios(iter, compression_ratios(iter, :) > 0));
    fitness_values(iter) = sum(Q, 'all') + mean(energy_levels) + sum(data_at_base(iter, :)) - coverage + avg_compression;
end

% Optimized Routing Path
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
        break;
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
ylabel('Fitness (Q-Values + Energy + Data + Compression - Coverage)');
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
scatter(base_station(1), base_station(2), 150, 'g^', 'filled');
grid on;
xlabel('X Coordinate (meters)');
ylabel('Y Coordinate (meters)');
title('Final Cluster Head Selection with Base Station');
legend('Nodes', 'Cluster Heads', 'Base Station');
hold off;

% 4. Optimized Routing Path
figure;
plot(nodes(route, 1), nodes(route, 2), '-s', 'LineWidth', 2, 'MarkerSize', 10);
grid on;
xlabel('X Coordinate (meters)');
ylabel('Y Coordinate (meters)');
title('Optimized Routing Path');
legend('Routing Path');

% 5. Data Received at Base Station
figure;
plot(1:max_iterations, sum(data_at_base, 2), '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
grid on;
xlabel('Iterations');
ylabel('Total Data Received');
title('Data Received at Base Station Over Iterations');
legend('Total Data');

% 6. Compression Ratio Over Iterations
figure;
plot(1:max_iterations, mean(compression_ratios, 2), '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
grid on;
xlabel('Iterations');
ylabel('Average Compression Ratio');
title('Compression Ratio Over Iterations');
legend('Compression Ratio');
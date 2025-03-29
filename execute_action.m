function [reward, next_node, energy_levels] = execute_action(current_node, action, energy_levels, threshold)
    % Simulate energy cost and reward
    energy_cost = rand * 10; % Random energy cost for action
    is_valid = energy_levels(action) > threshold; % Check energy threshold

    if is_valid
        reward = energy_levels(action) - energy_cost; % Positive reward
        energy_levels(action) = energy_levels(action) - energy_cost; % Update energy
        next_node = action;
    else
        reward = -1; % Negative reward for invalid action
        next_node = current_node; % Stay at the same node
    end
end

function route = rl_routing(Q, start_node, end_node, max_steps)
    route = start_node; % Initialize route
    current_node = start_node;
    visited_nodes = zeros(1, max_steps); % Track visited nodes to avoid infinite loops
    visited_nodes(1) = current_node; % Mark start node as visited
    steps = 1; % Number of steps taken

    while current_node ~= end_node
        [~, next_node] = max(Q(current_node, :)); % Choose the best action (greedy)
        
        % Check if the next node is already visited (to avoid loops)
        if any(visited_nodes(1:steps) == next_node)
            % Exploration: Choose a random unvisited node if stuck
            unvisited_nodes = setdiff(1:size(Q, 2), visited_nodes(1:steps));
            if ~isempty(unvisited_nodes)
                next_node = unvisited_nodes(randi(length(unvisited_nodes)));
            else
                warning('Exploration failed, trying to get out of loop');
                break; % Exit if no unvisited nodes are available
            end
        end

        route = [route, next_node]; % Add to route
        visited_nodes(steps + 1) = next_node; % Mark node as visited
        current_node = next_node; % Move to next node
        steps = steps + 1;

        % Prevent infinite loops if we reach max steps
        if steps > max_steps
            warning('Max steps reached, routing terminated!');
            break;
        end
    end
end

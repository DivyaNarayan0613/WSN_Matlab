function action = epsilon_greedy(Q, state, epsilon)
    if rand < epsilon
        action = randi(size(Q, 2)); % Random action (exploration)
    else
        [~, action] = max(Q(state, :)); % Best action (exploitation)
    end
end

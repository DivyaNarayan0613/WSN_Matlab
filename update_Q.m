function Q = update_Q(Q, P, beta, reward, next_state, alpha, gamma)
    max_next_Q = max(Q(next_state, :)); % Max Q-value for next state
    Q(P, beta) = (1 - alpha) * Q(P, beta) + alpha * (reward + gamma * max_next_Q);
end

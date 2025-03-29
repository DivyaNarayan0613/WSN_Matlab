function [next_state, reward, done, info] = step(action)
    % Simulate the environment's response to an action.
    % If no action is provided, default to action = 1.
    if nargin < 1
        action = 1;  % Default action
    end
    
    % Define environment parameters
    state_size = 10;  % Size of the state vector
    max_steps = 50;   % Maximum number of steps before the episode ends
    
    % Static variables to maintain step count
    persistent step_count;
    if isempty(step_count)
        step_count = 0;
    end
    
    % Increment the step count
    step_count = step_count + 1;
    
    % Generate the next state as a random vector
    next_state = rand(1, state_size);
    
    % Define reward based on the action
    reward = action * 10;  % Example: Linear relationship with action
    
    % Define terminal condition (episode ends after max_steps)
    done = (step_count >= max_steps);
    
    % Reset step count if the episode is done
    if done
        step_count = 0;  % Reset step counter for the next episode
    end
    
    % Additional information (optional)
    info = struct('action_taken', action, 'step_number', step_count);
end

classdef DQNAgent
    properties
        state_size
        action_size
        model
        replay_buffer
        batch_size = 32
        gamma = 0.99  % Discount factor
        epsilon = 1.0  % Exploration rate
        epsilon_min = 0.01
        epsilon_decay = 0.995
        learning_rate = 0.001
        optimizer
    end
    
    methods
        % Constructor
        function obj = DQNAgent(state_size, action_size)
            obj.state_size = state_size;
            obj.action_size = action_size;
            obj.model = obj.createModel();  % Create the DQN model
            obj.replay_buffer = {};  % Initialize replay buffer
            obj.optimizer = adamoptimizer('LearnRate', obj.learning_rate);
        end
        
        % Create DQN Model (simple fully connected network)
        function model = createModel(obj)
            layers = [
                featureInputLayer(obj.state_size)
                fullyConnectedLayer(24)
                reluLayer
                fullyConnectedLayer(24)
                reluLayer
                fullyConnectedLayer(obj.action_size)
            ];
            model = dlnetwork(layers);
        end
        
        % Select action using epsilon-greedy policy
        function action = selectAction(obj, state)
            if rand < obj.epsilon
                % Exploration: Random action
                action = randi([1, obj.action_size]);
            else
                % Exploitation: Select action with highest Q-value
                state_dl = dlarray(single(state));
                q_values = predict(obj.model, state_dl);
                [~, action] = max(q_values);
            end
        end
        
        % Store experience in the replay buffer
        function obj = storeExperience(obj, state, action, reward, next_state, done)
            experience = {state, action, reward, next_state, done};
            obj.replay_buffer{end+1} = experience;
        end
        
        % Sample a batch of experiences from the replay buffer
        function [state_batch, action_batch, reward_batch, next_state_batch, done_batch] = sampleExperience(obj)
            % Sample random indices
            indices = randperm(numel(obj.replay_buffer), obj.batch_size);
            
            state_batch = [];
            action_batch = [];
            reward_batch = [];
            next_state_batch = [];
            done_batch = [];
            
            for i = 1:obj.batch_size
                experience = obj.replay_buffer{indices(i)};
                state_batch(i, :) = experience{1};
                action_batch(i) = experience{2};
                reward_batch(i) = experience{3};
                next_state_batch(i, :) = experience{4};
                done_batch(i) = experience{5};
            end
        end
        
        % Train the DQN model using the sampled experience
        function obj = trainModel(obj, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            % Calculate Q-values
            state_dl = dlarray(single(state_batch));
            next_state_dl = dlarray(single(next_state_batch));
            
            % Get Q-values for current states and next states
            q_values = predict(obj.model, state_dl);
            next_q_values = predict(obj.model, next_state_dl);
            
            % Update target Q-values using Bellman equation
            target_q_values = q_values;
            for i = 1:obj.batch_size
                if done_batch(i)
                    target_q_values(i, action_batch(i)) = reward_batch(i);
                else
                    target_q_values(i, action_batch(i)) = reward_batch(i) + obj.gamma * max(next_q_values(i, :));
                end
            end
            
            % Compute loss
            loss = mean((q_values - target_q_values).^2);
            
            % Compute gradients and update the model
            gradients = dlgradient(loss, obj.model.Learnables);
            obj.model = obj.optimizer(obj.model, gradients);
            
            % Decay epsilon
            obj.epsilon = max(obj.epsilon * obj.epsilon_decay, obj.epsilon_min);
        end
    end
end

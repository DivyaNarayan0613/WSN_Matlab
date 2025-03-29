% --- Deep Q-Learning for Routing Protocol in Wireless Sensor Networks ---
% Initialize parameters
numNodes = 10;  % Number of nodes in the network (adjust based on your network size)
stateSize = numNodes;  % Example state space size (e.g., node energy, distance)
actionSize = numNodes;  % Action space size (routing to a particular node)
gamma = 0.99;  % Discount factor (for future rewards)
epsilon = 1.0;  % Exploration rate (epsilon-greedy)
epsilonDecay = 0.995;  % Decay rate for epsilon
epsilonMin = 0.01;  % Minimum epsilon
alpha = 0.01;  % Learning rate
maxEpisodes = 1000;  % Number of training episodes
maxSteps = 100;  % Maximum steps per episode
replayBufferSize = 10000;  % Replay buffer size
batchSize = 64;  % Batch size for training

% Initialize the replay buffer
replayBuffer = {};

% Define the neural network architecture for Q-function approximation
layers = [
    imageInputLayer([stateSize, 1, 1], 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(actionSize, 'Name', 'fc3')  % Output Q-values for each action
    regressionLayer('Name', 'output')
];

% Specify options for training the neural network
options = trainingOptions('adam', ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', batchSize, ...
    'InitialLearnRate', alpha, ...
    'Verbose', false);

% Epsilon-Greedy Action Selection
function action = chooseAction(state, Qnetwork, epsilon)
    if rand < epsilon
        % Exploration: choose a random action
        action = randi(actionSize);  % Random action
    else
        % Exploitation: choose action with the max Q-value
        Q_values = predict(Qnetwork, state);
        [~, action] = max(Q_values);
    end
end

% Store experiences in the replay buffer
function addExperience(state, action, reward, nextState)
    experience = struct('state', state, 'action', action, 'reward', reward, 'nextState', nextState);
    replayBuffer{end + 1} = experience;
    if length(replayBuffer) > replayBufferSize
        replayBuffer(1) = [];  % Remove oldest experience to maintain the buffer size
    end
end

% Sample a batch from the replay buffer
function [batchState, batchAction, batchReward, batchNextState] = sampleBatch(batchSize)
    indices = randi([1, length(replayBuffer)], [batchSize, 1]);
    batchState = [];
    batchAction = [];
    batchReward = [];
    batchNextState = [];
    
    for i = 1:batchSize
        experience = replayBuffer{indices(i)};
        batchState = [batchState, experience.state];
        batchAction = [batchAction, experience.action];
        batchReward = [batchReward, experience.reward];
        batchNextState = [batchNextState, experience.nextState];
    end
end

% Q-learning update using Deep Q-Network
function trainDQN(Qnetwork, batchState, batchAction, batchReward, batchNextState, gamma)
    % Predict current Q-values for batch
    currentQ = predict(Qnetwork, batchState);
    
    % Predict next Q-values for batch
    nextQ = predict(Qnetwork, batchNextState);
    
    % Compute the target Q-values (Bellman equation)
    targetQ = currentQ;
    for i = 1:length(batchReward)
        action = batchAction(i);
        targetQ(i, action) = batchReward(i) + gamma * max(nextQ(i, :));
    end
    
    % Train the network (backpropagation)
    Qnetwork = trainNetwork(batchState, targetQ, layers, options);
end

% --- Main Training Loop ---
% Initialize Q-network with a dummy batch (start with a simple state)
dummyState = zeros(stateSize, 1, batchSize);  % Create dummy state with batchSize
dummyAction = zeros(actionSize,1, batchSize);  % Create dummy action for each batch
Qnetwork = trainNetwork(dummyState, dummyAction, layers, options);

% Training loop
for episode = 1:maxEpisodes
    state = rand(stateSize, 1);  % Example: Random initial state (can be set based on your network)
    
    totalReward = 0;
    
    for step = 1:maxSteps
        % Choose an action based on the current state
        action = chooseAction(state, Qnetwork, epsilon);
        
        % Take action and observe next state and reward
        % (This part depends on your specific environment setup)
        nextState = rand(stateSize, 1);  % Example: Random next state (replace with actual model)
        reward = rand();  % Example: Random reward (replace with actual reward)
        
        % Add experience to the replay buffer
        addExperience(state, action, reward, nextState);
        
        % Sample a batch of experiences from the replay buffer
        [batchState, batchAction, batchReward, batchNextState] = sampleBatch(batchSize);
        
        % Train the model using the sampled batch
        trainDQN(Qnetwork, batchState, batchAction, batchReward, batchNextState, gamma);
        
        % Update total reward and state
        totalReward = totalReward + reward;
        state = nextState;
        
        % Decay epsilon for exploration
        epsilon = max(epsilon * epsilonDecay, epsilonMin);
    end
    
    % Display progress
    disp(['Episode ', num2str(episode), ' Total Reward: ', num2str(totalReward)]);
end

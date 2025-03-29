function model = createDQN(state_size, action_size)
    layers = [
        featureInputLayer(state_size, 'Normalization', 'none', 'Name', 'state')  % Input layer for state
        fullyConnectedLayer(64, 'Name', 'fc1')  % First fully connected layer
        reluLayer('Name', 'relu1')  % ReLU activation function
        fullyConnectedLayer(64, 'Name', 'fc2')  % Second fully connected layer
        reluLayer('Name', 'relu2')  % ReLU activation function
        fullyConnectedLayer(action_size, 'Name', 'output')  % Output layer for Q-values, one for each action
    ];

    % Create the dlnetwork object
    model = dlnetwork(layers);
end

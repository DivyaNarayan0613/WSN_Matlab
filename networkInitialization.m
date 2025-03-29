function [nodePositions, nodeEnergy] = networkInitialization(numNodes, areaSize, energyInit)
    % Generate random node positions and initialize energy levels
    nodePositions = rand(numNodes, 2) .* areaSize;
    nodeEnergy = ones(numNodes, 1) * energyInit;
    
    % Visualize network
    figure;
    scatter(nodePositions(:,1), nodePositions(:,2), 'filled');
    title('Node Positions in the Network');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    grid on;
end

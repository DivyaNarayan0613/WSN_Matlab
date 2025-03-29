function [bestRoute, energyCost] = energyAwareRouting(nodePositions, nodeEnergy, sourceNode, targetNode, commRange)
    bestRoute = sourceNode;
    energyCost = 0;
    visited = false(length(nodeEnergy), 1);
    visited(sourceNode) = true;
    currentNode = sourceNode;

    while currentNode ~= targetNode
        % Get neighbors
        neighbors = findNeighbors(nodePositions, currentNode, commRange);
        minEnergy = inf;
        nextNode = -1;

        for i = 1:length(neighbors)
            if ~visited(neighbors(i))
                energyRequired = calculateEnergy(nodePositions(currentNode,:), nodePositions(neighbors(i),:), nodeEnergy(currentNode));
                if energyRequired < minEnergy
                    minEnergy = energyRequired;
                    nextNode = neighbors(i);
                end
            end
        end

        if nextNode == -1
            disp('No valid route found');
            return;
        end

        bestRoute = [bestRoute, nextNode];
        energyCost = energyCost + minEnergy;
        visited(nextNode) = true;
        currentNode = nextNode;
    end
end

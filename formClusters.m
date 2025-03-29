function [clusterHeads, nodeClusters] = formClusters(nodePositions, nodeEnergy, numNodes, commRange)
    k = 10; % Number of cluster heads
    [~, idx] = sort(nodeEnergy, 'descend'); % Sort nodes by energy
    clusterHeads = idx(1:k); % Top k nodes as cluster heads
    
    % Assign nodes to nearest cluster heads
    nodeClusters = zeros(numNodes, 1);
    for i = 1:numNodes
        if ~ismember(i, clusterHeads)
            minDistance = inf;
            for j = 1:k
                distance = norm(nodePositions(i,:) - nodePositions(clusterHeads(j),:));
                if distance <= commRange && distance < minDistance
                    minDistance = distance;
                    nodeClusters(i) = clusterHeads(j);
                end
            end
        end
    end
end

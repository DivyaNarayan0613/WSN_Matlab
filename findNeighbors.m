function neighbors = findNeighbors(nodePositions, currentNode, commRange)
    neighbors = [];
    for i = 1:size(nodePositions, 1)
        if i ~= currentNode && norm(nodePositions(currentNode,:) - nodePositions(i,:)) <= commRange
            neighbors = [neighbors, i];
        end
    end
end

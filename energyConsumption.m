function energyLeft = energyConsumption(energy, distance)
    % Calculate energy consumption for given distance
    energyPerDistance = 0.01; % Energy consumed per unit distance
    energyLeft = energy - energyPerDistance * distance;
end

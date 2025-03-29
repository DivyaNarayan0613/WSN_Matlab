function energyRequired = calculateEnergy(sourcePos, targetPos, energy)
    energyPerDistance = 0.01;
    distance = norm(sourcePos - targetPos);
    energyRequired = energyPerDistance * distance;
end

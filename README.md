This MATLAB program simulates a Reinforcement Learning (RL)-based Cluster Head Selection and Routing Protocol in a wireless sensor network (WSN). 

    The program first clears any previous variables and adds helper functions (stored in the 'functions' folder).

    Network Parameters are set:

        num_nodes = 50: The network consists of 50 nodes.

        max_iterations = 100: The learning process runs for 100 iterations.

        alpha = 0.1: Learning rate (controls how much new information overrides old knowledge).

        gamma = 0.9: Discount factor (determines how much future rewards matter).

        epsilon = 0.2: Exploration rate (20% chance of trying a random action instead of the best-known action).

        k = 1: Used for softmax exploration (not directly used here).

2. Setting Up Q-Learning

    States and Actions:

        Each node in the network is treated as a state.

        Each node can select another node as a cluster head (actions).

        Q = zeros(num_states, num_actions): Initializes a Q-table to store the best action for each node.

    Energy Levels:

        Each node is assigned a random energy level between 0 and 100.

        If energy falls below 20, the node is considered low on energy.

3. Learning Process (Simulation Loop)

    The program runs 100 learning iterations, where:

        Action Selection:

            The node picks a cluster head using the ε-greedy policy (epsilon_greedy function).

            80% of the time, it picks the best-known action (from the Q-table).

            20% of the time, it explores a random action to discover better options.

        Action Execution:

            The node executes the action (i.e., tries to join a cluster).

            The function execute_action returns:

                Reward: A score based on the energy levels.

                Next Node: The updated state after the action.

                Updated Energy Levels: Adjusts energy consumption.

        Q-Table Update:

            The Q-table is updated using the Q-learning formula:
            Q(s,a)=Q(s,a)+α×[reward+γ×max⁡Q(s′,a′)−Q(s,a)]
            Q(s,a)=Q(s,a)+α×[reward+γ×maxQ(s′,a′)−Q(s,a)]

            This helps the system learn the best cluster heads over time.

4. Identifying the Best Cluster Heads

    After training, each node selects its best cluster head based on the Q-table.

    The program displays the unique cluster heads chosen by the nodes.

5. Routing Example

    A routing example is simulated where data is sent from Node 30 to Node 50.

    The function rl_routing(Q, start_node, end_node, max_steps) finds the optimal route using the learned Q-values.

    The best route is displayed.

Summary

✅ This program learns the best cluster heads in a WSN using Reinforcement Learning.
✅ It ensures energy-efficient clustering and optimal routing based on Q-learning.
✅ The nodes gradually improve their decisions to maximize network lifetime.

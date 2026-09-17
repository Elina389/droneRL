"""
Consensus dynamics and graph Laplacian coordination for swarm intelligence.

This module implements the mathematical framework from the handwritten notes:
1. Adjacency matrix computation (who can see whom)
2. Graph Laplacian matrix (L = D - A)
3. Consensus dynamics (ẋ = -Lx)
4. Algebraic connectivity (λ2) for convergence analysis
5. Formation control and separation forces
6. Shape agreement and coordinated movement

Key equations implemented:
- aij = 1 if ||xi - xj|| < r (adjacency)
- ẋi = -∑j aij(xi - xj) (consensus)
- ẋ = -Lx, L = D - A (matrix form)
- ||x(t) - x̄|| ≤ e^(-λ2t)||x(0) - x̄|| (convergence)
- Separation force: ui^sep = ∑(dij<rs) ks(rs - dij)(xi - xj)/dij
"""
import numpy as np
from scipy.linalg import eigh


class ConsensusCoordinator:
    """
    Matrix-based swarm coordination using consensus dynamics and graph theory.
    
    This coordinator computes forces that make drones:
    1. Converge to consensus (agreement) positions
    2. Maintain safe separation distances
    3. Form and maintain specific shapes/formations
    4. Move the entire shape as one coordinated unit
    """
    
    def __init__(self, sensing_range=4.0, separation_range=2.0, 
                 separation_gain=1.5, consensus_gain=0.8, 
                 formation_gain=0.5, min_connectivity=0.1):
        """
        Parameters from the handwritten notes:
        
        sensing_range (r): Communication/sensing radius for adjacency matrix
        separation_range (rs): Safe distance threshold for separation forces
        separation_gain (ks): Push strength for separation (tunable k parameter)
        consensus_gain: Strength of consensus attraction forces
        formation_gain: Strength of shape maintenance forces  
        min_connectivity: Minimum λ2 required for convergence guarantee
        """
        self.sensing_range = sensing_range
        self.separation_range = separation_range
        self.separation_gain = separation_gain
        self.consensus_gain = consensus_gain
        self.formation_gain = formation_gain
        self.min_connectivity = min_connectivity
        
        # Formation control parameters
        self.target_formation = None  # δi values (desired positions in formation)
        self.formation_center = np.array([0.0, 0.0])  # c(t)
        self.formation_rotation = 0.0  # ωt for R(ωt)
        self.rotation_speed = 0.1  # ω
        
        # Metrics for analysis
        self.last_laplacian = None
        self.last_eigenvalues = None
        self.last_connectivity = 0.0
        
    def compute_adjacency_matrix(self, positions):
        """
        Build adjacency matrix A where aij = 1 if ||xi - xj|| < r
        
        From notes: "this matrix encodes the adjacency information (0 and 1)"
        """
        positions = np.array(positions)
        n = len(positions)
        A = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.sensing_range:
                        A[i, j] = 1.0
        
        return A
    
    def compute_laplacian_matrix(self, adjacency_matrix):
        """
        Compute graph Laplacian L = D - A
        
        From notes: "L is Laplacian of the graph"
        D = degree matrix (diagonal with number of neighbors)
        A = adjacency matrix  
        """
        A = adjacency_matrix
        # Degree matrix: Dii = number of neighbors of drone i
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        
        # Laplacian matrix
        L = D - A
        self.last_laplacian = L
        
        return L
    
    def compute_algebraic_connectivity(self, laplacian_matrix):
        """
        Compute λ2 (second-smallest eigenvalue) of Laplacian matrix.
        
        From notes: "λ2 > 0 ⟺ graph is connected"
        "λ2 > 0 ⟹ e^(-λ2t)||x(0) - x̄||" (convergence rate)
        
        Returns λ2 and all eigenvalues for analysis.
        """
        eigenvalues, _ = eigh(laplacian_matrix)
        eigenvalues = np.sort(eigenvalues)
        
        # First eigenvalue should be ~0 (within numerical precision)
        # Second eigenvalue is the algebraic connectivity
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        self.last_eigenvalues = eigenvalues
        self.last_connectivity = lambda_2
        
        return lambda_2, eigenvalues
    
    def consensus_forces(self, positions):
        """
        Compute consensus dynamics forces: ẋi = -∑j aij(xi - xj)
        
        From notes: "ẋi = -∑j aij(xi - xj)" and "ẋ = -Lx"
        This makes each drone move toward the average of its neighbors.
        """
        positions = np.array(positions)
        n = len(positions)
        
        if n <= 1:
            return np.zeros_like(positions)
        
        A = self.compute_adjacency_matrix(positions)
        forces = np.zeros_like(positions)
        
        # Individual consensus force for each drone
        for i in range(n):
            consensus_force = np.zeros(2)
            for j in range(n):
                if A[i, j] > 0:  # j is a neighbor of i
                    # Force toward neighbor j
                    consensus_force -= A[i, j] * (positions[i] - positions[j])
            forces[i] = self.consensus_gain * consensus_force
        
        return forces
    
    def separation_forces(self, positions):
        """
        Compute separation forces: ui^sep = ∑(dij<rs) ks(rs - dij)(xi - xj)/dij
        
        From notes: "k is variable that could be tuned to control how much 
        that push again each. we are simulating force field"
        
        Prevents drones from getting too close to each other.
        """
        positions = np.array(positions)
        n = len(positions)
        forces = np.zeros_like(positions)
        
        for i in range(n):
            separation_force = np.zeros(2)
            for j in range(n):
                if i != j:
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    
                    if 0 < dist < self.separation_range:
                        # Push away with strength proportional to proximity
                        direction = diff / dist  # unit vector
                        magnitude = self.separation_gain * (self.separation_range - dist) / dist
                        separation_force += magnitude * direction
            
            forces[i] = separation_force
        
        return forces
    
    def formation_forces(self, positions):
        """
        Compute formation control forces for shape maintenance.
        
        From notes: "ẋi = -∑j aij[(xi - xj) - (δi - δj)]"
        where δi is the desired position of drone i in the formation.
        
        Also implements moving formations: xi*(t) = c(t) + R(ωt)δi
        """
        if self.target_formation is None:
            return np.zeros_like(positions)
        
        positions = np.array(positions)
        n = len(positions)
        forces = np.zeros_like(positions)
        
        # Update formation center and rotation
        self.formation_rotation += self.rotation_speed
        rotation_matrix = np.array([
            [np.cos(self.formation_rotation), -np.sin(self.formation_rotation)],
            [np.sin(self.formation_rotation), np.cos(self.formation_rotation)]
        ])
        
        A = self.compute_adjacency_matrix(positions)
        
        for i in range(n):
            formation_force = np.zeros(2)
            
            if i < len(self.target_formation):
                # Desired position in rotated formation
                desired_relative_pos = rotation_matrix @ self.target_formation[i]
                desired_absolute_pos = self.formation_center + desired_relative_pos
                
                for j in range(n):
                    if A[i, j] > 0 and j < len(self.target_formation):
                        # Actual relative position difference
                        actual_diff = positions[i] - positions[j]
                        
                        # Desired relative position difference
                        desired_diff = (rotation_matrix @ self.target_formation[i] - 
                                      rotation_matrix @ self.target_formation[j])
                        
                        # Formation control force
                        formation_force -= A[i, j] * (actual_diff - desired_diff)
            
            forces[i] = self.formation_gain * formation_force
        
        return forces
    
    def compute_coordination_forces(self, positions):
        """
        Combine all coordination forces: consensus + separation + formation.
        
        Returns the complete force vector for each drone plus diagnostic info.
        """
        positions = np.array(positions)
        
        # Individual force components
        consensus = self.consensus_forces(positions)
        separation = self.separation_forces(positions)
        formation = self.formation_forces(positions)
        
        # Total coordination force
        total_forces = consensus + separation + formation
        
        # Compute diagnostics
        A = self.compute_adjacency_matrix(positions)
        L = self.compute_laplacian_matrix(A)
        lambda_2, eigenvalues = self.compute_algebraic_connectivity(L)
        
        # Check if swarm is well-connected
        is_connected = lambda_2 > self.min_connectivity
        
        diagnostics = {
            'adjacency_matrix': A,
            'laplacian_matrix': L,
            'eigenvalues': eigenvalues,
            'algebraic_connectivity': lambda_2,
            'is_connected': is_connected,
            'consensus_forces': consensus,
            'separation_forces': separation,
            'formation_forces': formation,
            'convergence_rate': lambda_2,  # Rate of exponential convergence
        }
        
        return total_forces, diagnostics
    
    def set_formation(self, formation_positions, center=None):
        """
        Set target formation shape (δi values for each drone).
        
        formation_positions: List of (x, y) relative positions for each drone
        center: Formation center point (defaults to current average position)
        """
        self.target_formation = np.array(formation_positions)
        if center is not None:
            self.formation_center = np.array(center)
    
    def predict_convergence_time(self, positions, tolerance=0.1):
        """
        Predict how long it will take for swarm to converge to consensus.
        
        From notes: "||x(t) - x̄|| ≤ e^(-λ2t)||x(0) - x̄||"
        
        Returns estimated time for swarm spread to reduce to tolerance level.
        """
        if self.last_connectivity <= 0:
            return float('inf')  # Won't converge if not connected
        
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        initial_spread = np.max([np.linalg.norm(pos - center) for pos in positions])
        
        if initial_spread < tolerance:
            return 0.0  # Already converged
        
        # Solve: tolerance = initial_spread * e^(-λ2 * t)
        # t = ln(initial_spread / tolerance) / λ2
        convergence_time = np.log(initial_spread / tolerance) / self.last_connectivity
        
        return convergence_time


def snap_consensus_to_discrete_actions(consensus_forces, env, positions, stay_threshold=0.1):
    """
    Convert continuous consensus forces to discrete environment actions.
    
    This bridges the gap between the continuous mathematical framework 
    and the discrete action space of the grid environment.
    """
    actions = {}
    action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, stay
    
    for i, agent in enumerate(env.agents):
        if i >= len(consensus_forces):
            actions[agent] = 4  # stay
            continue
        
        force = consensus_forces[i]
        force_magnitude = np.linalg.norm(force)
        
        if force_magnitude < stay_threshold:
            actions[agent] = 4  # stay
            continue
        
        # Find best matching discrete action
        best_action = 4
        best_score = -np.inf
        
        pos = positions[i] if i < len(positions) else env.positions[agent]
        
        for action_idx, (dr, dc) in enumerate(action_deltas[:4]):  # exclude stay
            # Check if action is legal (not blocked)
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (new_pos[0] < 0 or new_pos[0] >= env.grid_size or 
                new_pos[1] < 0 or new_pos[1] >= env.grid_size or 
                new_pos in env.obstacles):
                continue
            
            # Compute alignment with consensus force
            action_vector = np.array([dr, dc])
            score = np.dot(force, action_vector)
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        actions[agent] = best_action
    
    return actions
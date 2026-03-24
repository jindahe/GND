import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt

def create_local_weights(p):
    """
    Creates the edge weights for the classical probability distribution.
    0: no error (probability 1-p)
    1: error (probability p)
    """
    return np.diag([1 - p, p])

def build_anyon_tn(L, p):
    """
    Builds the 2D tensor network representing the unnormalized anyon 
    probability distribution (RBIM partition function) on an L x L grid.
    """
    # Create the edge weight matrix
    W = create_local_weights(p)
    
    # Square root of weights to distribute symmetrically on bonds
    W_sqrt = np.sqrt(W)
    
    # Delta tensor enforces the classical Z2 parity rule at each vertex:
    # nonzero when i XOR j XOR k XOR l == 0 (even parity)
    delta = np.zeros((2, 2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i ^ j ^ k ^ l) == 0:
                        m = i ^ j ^ k ^ l
                        delta[i, j, k, l] = 1.0
    
    # Absorb weights into the vertex tensors [up, right, down, left]
    # In a rigorous RBIM mapping, we apply W_sqrt to each leg
    local_tensor = np.einsum('ijklm,ia,jb,kc,ld->abcdm',
                             delta, W_sqrt, W_sqrt, W_sqrt, W_sqrt)

    # Build 2D TN manually with periodic boundary conditions.
    # Bond index naming:
    #   vertical bond below site (i,j):   v{i},{j}  (= up bond of site (i+1,j))
    #   horizontal bond right of site (i,j): h{i},{j} (= left bond of site (i,j+1))
    tensors = []
    for i in range(L):
        for j in range(L):
            inds = (
                f'v{(i-1)%L},{j}',  # up   (shared with site above)
                f'h{i},{j}',         # right (shared with site to the right)
                f'v{i},{j}',         # down  (shared with site below)
                f'h{i},{(j-1)%L}',  # left  (shared with site to the left)
                f'phys_{i}_{j}'         # physical index for this site
            )
            tags = {f'I{i},J{j}', f'ROW{i}', f'COL{j}'}
            tensors.append(qtn.Tensor(local_tensor.copy(), inds=inds, tags=tags))

    return qtn.TensorNetwork(tensors)

def get_region_tags(L, r_width):
    """
    Defines the spatial tags for Regions A, B, and C.
    A is the central 2x2 area.
    B is the annulus of width r_width around A.
    C is everything else.
    """
    center = L // 2
    A_coords = [(center, center), (center-1, center), 
                (center, center-1), (center-1, center-1)]
    
    tags_A, tags_B, tags_C = [], [], []
    
    for i in range(L):
        for j in range(L):
            tag = f'I{i},J{j}'
            dist_i = min(abs(i - center), abs(i - (center-1)))
            dist_j = min(abs(j - center), abs(j - (center-1)))
            max_dist = max(dist_i, dist_j)
            
            if (i, j) in A_coords:
                tags_A.append(tag)
            elif max_dist <= r_width:
                tags_B.append(tag)
            else:
                tags_C.append(tag)
                
    return tags_A, tags_B, tags_C

def compute_shannon_entropy(tn, keep_tags, env_tags, max_bond_dim=32):
    """
    Contracts the environment (env_tags) using BMPS, leaving the keep_tags open.
    Calculates the Shannon entropy of the resulting normalized probability tensor.
    """
    tn_calc = tn.copy()
    keep_phys_inds = []
    for tag in keep_tags:
        # 解析 tag 'Ii,Jj' 提取 i 和 j
        parts = tag.replace('I', '').replace('J', '').split(',')
        i, j = parts, parts
        keep_phys_inds.append(f'phys_{i}_{j}')

    if env_tags:
        # Contract only the environment tensors, leaving keep_tags open
        tn_calc.contract_tags(env_tags, optimize='auto', inplace=True)

    # Contract the remaining (kept) region exactly
    reduced_tensor = tn_calc.contract(output_inds=keep_phys_inds, optimize='auto')
    
    # Flatten to a 1D probability array (convert memoryview to ndarray if needed)
    probs = np.array(reduced_tensor.data).flatten()
    
    # Normalize the probabilities
    probs = probs / np.sum(probs)
    
    # Remove zeros to avoid log(0)
    probs = probs[probs > 1e-12]
    
    # Shannon Entropy: - \sum P log(P)
    # Note: The paper uses log base 2 for quantum information quantities
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy

def calculate_cmi(L, p, r_width, max_bond_dim=32):
    """
    Calculates I(A:C|B) = H(AB) + H(BC) - H(B) - H(ABC)
    *Note: In this specific mapping, H(ABC) is the entropy of the whole system.
    """
    tn = build_anyon_tn(L, p)
    tags_A, tags_B, tags_C = get_region_tags(L, r_width)
    
    # Calculate required entropies
    # To rigorously handle pi(m_A) for H(BC) and H(B), we compute the entropy 
    # over the combined regions and apply boundary parity projectors. 
    # For this script, we approximate the joint entropies directly via spatial partition.
    
    H_AB = compute_shannon_entropy(tn, tags_A + tags_B, tags_C, max_bond_dim)
    H_BC = compute_shannon_entropy(tn, tags_B + tags_C, tags_A, max_bond_dim)
    H_B  = compute_shannon_entropy(tn, tags_B, tags_A + tags_C, max_bond_dim)
    H_ABC = compute_shannon_entropy(tn, tags_A + tags_B + tags_C, [], max_bond_dim)
    
    cmi = H_AB + H_BC - H_B - H_ABC
    return cmi

def run_scaling_experiment():
    L = 10 # System size (L x L)
    r_values = [1, 2, 3] # Buffer widths
    p_values = [0.05, 0.11, 0.15] # 0.05 (Topological), 0.11 (Critical), 0.15 (Trivial)
    
    results = {p: [] for p in p_values}
    
    for p in p_values:
        print(f"--- Simulating p = {p} ---")
        for r in r_values:
            cmi = calculate_cmi(L, p, r, max_bond_dim=16) # Lower max_bond for speed in demo
            results[p].append(cmi)
            print(f"  r = {r}: CMI = {cmi:.4f}")
            
    plot_results(r_values, results)

def plot_results(r_values, results):
    plt.figure(figsize=(8, 6))
    for p, cmi_vals in results.items():
        if p == 0.11:
            plt.plot(r_values, cmi_vals, 'o--', label=f'p = {p} (Critical)', color='orange')
        else:
            plt.plot(r_values, cmi_vals, 'o--', label=f'p = {p}')
            
    plt.yscale('log')
    plt.xlabel('Buffer Width (r)')
    plt.ylabel('Conditional Mutual Information $I(A:C|B)$')
    plt.title('Scaling of CMI in Dephased 2D Toric Code')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    run_scaling_experiment()
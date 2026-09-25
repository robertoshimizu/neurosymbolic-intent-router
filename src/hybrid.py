import numpy as np


# =====================================================================
# 1. KNOWLEDGE-BASED SYSTEM (KBS / RULE ENGINE)
# =====================================================================
class KnowledgeBaseEngine:
    """Deterministic symbolic engine that enforces business logic and safety rules."""

    def evaluate_permissions(
        self, user_context: dict, candidate_actions: list[str]
    ) -> dict[str, bool]:
        """Evaluates system facts and returns a boolean permission mask for candidates."""
        permissions = {}

        is_auth = user_context.get("is_authenticated", False)
        role = user_context.get("role", "guest")
        balance = user_context.get("account_balance", 0.0)

        for action in candidate_actions:
            # Rule 1: Admin actions require 'admin' role
            if action in ["delete_account", "override_limits"]:
                permissions[action] = is_auth and (role == "admin")

            # Rule 2: Financial transfers require auth, active status, and positive balance
            elif action in ["wire_transfer_funds", "issue_refund"]:
                permissions[action] = (
                    is_auth
                    and (user_context.get("status") == "active")
                    and balance > 0
                )

            # Rule 3: Information viewing
            elif action in ["view_account_balance", "view_public_faq"]:
                permissions[action] = is_auth or (action == "view_public_faq")

            else:
                permissions[action] = False

        return permissions


# =====================================================================
# 2. SYSTEM 1 CONTRASTIVE CLASSIFIER
# =====================================================================
class System1ContrastiveClassifier:
    """Sub-millisecond vector matcher using pre-cached candidate embeddings."""

    def __init__(self, action_embeddings: dict[str, np.ndarray]):
        # Normalize pre-cached action embeddings
        self.cached_actions = {
            action_id: vec / np.linalg.norm(vec)
            for action_id, vec in action_embeddings.items()
        }

    def compute_raw_similarity(
        self, state_vector: np.ndarray
    ) -> dict[str, float]:
        """Calculates cosine similarity (dot product) between state and action vectors."""
        state_norm = state_vector / np.linalg.norm(state_vector)
        return {
            action_id: float(np.dot(state_norm, action_vec))
            for action_id, action_vec in self.cached_actions.items()
        }


# =====================================================================
# 3. NEURO-SYMBOLIC PIPELINE (COMPOSITION)
# =====================================================================
def execute_neuro_symbolic_pipeline(
    user_query: str,
    state_vector: np.ndarray,
    user_context: dict,
    classifier: System1ContrastiveClassifier,
    kbs: KnowledgeBaseEngine,
):
    # Step 1: System 1 Neural Match (High Recall - calculates semantic similarity)
    raw_scores = classifier.compute_raw_similarity(state_vector)

    # Step 2: KBS Symbolic Evaluation (High Precision - checks logical rules)
    candidate_ids = list(raw_scores.keys())
    permissions = kbs.evaluate_permissions(user_context, candidate_ids)

    # Step 3: Apply Dynamic Masking
    # Illegal actions are masked with -infinity (-1e9) before softmax/selection
    masked_scores = {}
    for action_id, score in raw_scores.items():
        if permissions[action_id]:
            masked_scores[action_id] = score
        else:
            masked_scores[action_id] = (
                -1e9
            )  # Hard mask: eliminates illegal action

    # Step 4: Softmax over masked logit scores
    scores_array = np.array(list(masked_scores.values()))
    exp_scores = np.exp(scores_array - np.max(scores_array))
    probabilities = exp_scores / np.sum(exp_scores)
    prob_dict = dict(zip(candidate_ids, probabilities))

    # Final Decision Selection
    selected_action = max(masked_scores, key=masked_scores.get)

    # Output Execution Summary
    print(f"\nUser Query: '{user_query}'")
    print(
        f"Context: Auth={user_context.get('is_authenticated')}, Role='{user_context.get('role')}', Balance=${user_context.get('account_balance')}"
    )
    print("\nAction Decision Matrix:")
    print(
        f"{'Candidate Action':<24} | {'Raw Similarity':<15} | {'Allowed?':<8} | {'Final Prob':<10}"
    )
    print("-" * 68)
    for act in candidate_ids:
        allowed_str = "YES" if permissions[act] else "NO"
        print(
            f"{act:<24} | {raw_scores[act]:<15.4f} | {allowed_str:<8} | {prob_dict[act]:<10.2%}"
        )

    print(f"\nSelected Executable Action: >>> {selected_action} <<<\n")


# =====================================================================
# 4. EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    # Pre-cached action vectors (representing candidate tools in vector space)
    cached_action_vectors = {
        "wire_transfer_funds": np.array([0.88, 0.15, 0.10]),
        "view_account_balance": np.array([0.72, 0.30, 0.12]),
        "delete_account": np.array([0.82, 0.10, 0.05]),
        "view_public_faq": np.array([0.10, 0.85, 0.35]),
    }

    classifier = System1ContrastiveClassifier(cached_action_vectors)
    kbs = KnowledgeBaseEngine()

    # User Query Vector: "I want to send $5,000 to my external bank account."
    # High semantic affinity to 'wire_transfer_funds'
    query_state_vector = np.array([0.90, 0.12, 0.08])

    # --- Scenario A: Guest User (Not Authenticated) ---
    # System 1 wants to pick 'wire_transfer_funds', but KBS blocks it.
    # System falls back to the highest semantically relevant ALLOWED action ('view_public_faq').
    guest_context = {
        "is_authenticated": False,
        "role": "guest",
        "account_balance": 0.0,
    }
    execute_neuro_symbolic_pipeline(
        "I want to send $5,000 to my external bank account.",
        query_state_vector,
        guest_context,
        classifier,
        kbs,
    )

    # --- Scenario B: Authenticated User with $0 Balance ---
    # KBS blocks wire transfer (insufficient funds), forcing selection of 'view_account_balance'.
    zero_bal_context = {
        "is_authenticated": True,
        "role": "customer",
        "status": "active",
        "account_balance": 0.0,
    }
    execute_neuro_symbolic_pipeline(
        "I want to send $5,000 to my external bank account.",
        query_state_vector,
        zero_bal_context,
        classifier,
        kbs,
    )

    # --- Scenario C: Authenticated User with Sufficient Funds ---
    # KBS approves, System 1 selects the optimal action 'wire_transfer_funds'.
    valid_context = {
        "is_authenticated": True,
        "role": "customer",
        "status": "active",
        "account_balance": 10000.0,
    }
    execute_neuro_symbolic_pipeline(
        "I want to send $5,000 to my external bank account.",
        query_state_vector,
        valid_context,
        classifier,
        kbs,
    )

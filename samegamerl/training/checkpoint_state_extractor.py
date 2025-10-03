"""State extraction pattern for checkpoint system.

Extracts checkpoint state from domain objects without modifying them.
This adapter pattern keeps DqnAgent and SameGameEnv focused on their
core responsibilities while enabling checkpoint persistence.
"""

from samegamerl.training.checkpoint_data import AgentCheckpointState, EnvCheckpointState


class CheckpointStateExtractor:
    """Extracts checkpoint state from agent and environment instances.

    Uses adapter pattern to decouple domain classes from checkpoint system.
    Instead of adding get_checkpoint_state() methods to DqnAgent and SameGameEnv,
    this extractor pulls the necessary state and also handles restoration.
    """

    def extract_agent_state(self, agent) -> AgentCheckpointState:
        """Extract agent hyperparameters and training state.

        Captures current epsilon value (which decays during training) and all
        other hyperparameters necessary for resuming training from checkpoint.

        Args:
            agent: DqnAgent instance

        Returns:
            AgentCheckpointState with all hyperparameters
        """
        # Extract replay buffer capacity from the deque's maxlen
        replay_buffer_capacity = agent.replay_buffer.buffer.maxlen

        return AgentCheckpointState(
            epsilon=agent.epsilon,
            epsilon_min=agent.epsilon_min,
            epsilon_decay=agent.epsilon_decay,
            learning_rate=agent.learning_rate,
            gamma=agent.gamma,
            tau=agent.tau,
            batch_size=agent.batch_size,
            replay_buffer_size=replay_buffer_capacity,
        )

    def extract_env_state(self, env) -> EnvCheckpointState:
        """Extract environment configuration and reward parameters.

        Captures parameterized reward function settings that may be tuned
        during training experiments.

        Args:
            env: SameGameEnv instance

        Returns:
            EnvCheckpointState with reward parameters and game config
        """
        return EnvCheckpointState(
            completion_reward=env.completion_reward,
            partial_completion_base=env.partial_completion_base,
            invalid_move_penalty=env.invalid_move_penalty,
            singles_reduction_weight=env.singles_reduction_weight,
            game_config=env.config,
        )

    def restore_agent_state(self, state: AgentCheckpointState, agent) -> None:
        """Restore agent hyperparameters from checkpoint state.

        Updates agent instance in-place with saved hyperparameters.
        Critical for resuming training with correct epsilon value (which decays)
        and other hyperparameters that may have been modified during training.

        Args:
            state: AgentCheckpointState to restore from
            agent: DqnAgent instance to update
        """
        agent.epsilon = state.epsilon
        agent.epsilon_min = state.epsilon_min
        agent.epsilon_decay = state.epsilon_decay
        agent.learning_rate = state.learning_rate
        agent.gamma = state.gamma
        agent.tau = state.tau
        agent.batch_size = state.batch_size

        # Update optimizer learning rate if it changed
        for param_group in agent.opt.param_groups:
            param_group["lr"] = state.learning_rate

"""Checkpoint service for creating and loading checkpoints.

Coordinates checkpoint creation by extracting state from components,
saving model weights, and persisting via repository.
"""

from datetime import datetime

from samegamerl.training.checkpoint_data import CheckpointData, TrainingState
from samegamerl.training.checkpoint_state_extractor import CheckpointStateExtractor


class CheckpointService:
    """Service for creating and loading checkpoints.

    Handles coordination between state extraction, model weight saving,
    and repository persistence. Keeps checkpoint creation logic separate
    from training orchestration.

    Model weights are stored via the repository's get_model_path() method,
    ensuring all checkpoint-related files are colocated.
    """

    def __init__(self, repository):
        """Initialize checkpoint service.

        Args:
            repository: Checkpoint repository (must have get_model_path() method)
        """
        self.repository = repository
        self.extractor = CheckpointStateExtractor()

    def create_checkpoint(
        self,
        agent,
        env,
        experiment_name: str,
        epoch: int,
        total_epochs: int,
        total_steps: int,
        loss_history: list[float],
        benchmark_results: dict | None = None,
        random_seed: int = 42,
        best_score: float | None = None,
        training_time_seconds: float | None = None,
    ) -> str:
        """Create a checkpoint snapshot of current training state.

        Extracts state from agent and environment, saves model weights,
        and persists complete checkpoint via repository.

        Args:
            agent: DqnAgent instance
            env: SameGameEnv instance
            experiment_name: Name of the experiment
            epoch: Current epoch number
            total_epochs: Total epochs planned for training
            total_steps: Total steps taken so far
            loss_history: Recent loss values
            benchmark_results: Optional benchmark metrics
            random_seed: Random seed used for training
            best_score: Optional best score achieved
            training_time_seconds: Optional elapsed training time

        Returns:
            Checkpoint identifier string
        """
        # Extract component states
        agent_state = self.extractor.extract_agent_state(agent)
        env_state = self.extractor.extract_env_state(env)

        # Create training state
        training_state = TrainingState(
            total_epochs=total_epochs,
            current_epoch=epoch,
            total_steps=total_steps,
            random_seed=random_seed,
            best_score=best_score,
            training_time_seconds=training_time_seconds,
        )

        # Generate checkpoint ID to determine model path
        checkpoint_id = f"{experiment_name}_epoch_{epoch}"
        model_path = self.repository.get_model_path(checkpoint_id)
        model_filename = model_path.name

        # Save model weights via repository's path
        # Agent.save() adds .pth extension and uses agent.models_dir
        # Temporarily override to use repository's checkpoint directory
        original_models_dir = agent.models_dir
        agent.models_dir = model_path.parent

        # Save with name (without .pth extension, agent.save adds it)
        model_name_without_ext = model_path.stem
        agent.save(model_name_without_ext)

        # Restore original models_dir
        agent.models_dir = original_models_dir

        # Create checkpoint data
        checkpoint = CheckpointData(
            version=1,
            experiment_name=experiment_name,
            epoch=epoch,
            timestamp=datetime.now(),
            model_weights_filename=model_filename,
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=loss_history,
            benchmark_results=benchmark_results,
        )

        # Save checkpoint via repository
        checkpoint_id = self.repository.save(checkpoint)

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointData:
        """Load checkpoint from repository.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            CheckpointData loaded from repository
        """
        return self.repository.load(checkpoint_id)

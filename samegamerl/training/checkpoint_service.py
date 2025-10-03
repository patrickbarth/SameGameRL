"""Checkpoint service for creating and loading checkpoints.

Coordinates checkpoint creation by extracting state from components,
saving model weights, and persisting via repository.
"""

from datetime import datetime
from pathlib import Path

from samegamerl.training.checkpoint_data import CheckpointData, TrainingState
from samegamerl.training.checkpoint_state_extractor import CheckpointStateExtractor


class CheckpointService:
    """Service for creating and loading checkpoints.

    Handles coordination between state extraction, model weight saving,
    and repository persistence. Keeps checkpoint creation logic separate
    from training orchestration.
    """

    def __init__(self, repository, model_dir: Path | str):
        """Initialize checkpoint service.

        Args:
            repository: Checkpoint repository (pickle or database)
            model_dir: Directory for storing model weight files
        """
        self.repository = repository
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
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

        # Save model weights
        # Agent.save() adds .pth extension and uses agent.models_dir
        # We need to temporarily override models_dir to use our checkpoint directory
        original_models_dir = agent.models_dir
        agent.models_dir = self.model_dir

        model_name = f"{experiment_name}_epoch_{epoch}_model"
        model_filename = f"{model_name}.pth"
        agent.save(model_name)

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

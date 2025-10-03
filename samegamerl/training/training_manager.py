"""Training manager for coordinating training with optional checkpointing.

Provides a clean interface for training with checkpoint support while
keeping the existing train() function unchanged.
"""

from samegamerl.training.train import train


class TrainingOrchestrator:
    """Wraps the training loop for coordination.

    Delegates to the existing train() function without modification.
    Keeps training logic separate from checkpoint coordination.
    """

    def __init__(self, agent, env):
        """Initialize orchestrator with agent and environment.

        Args:
            agent: DqnAgent instance
            env: SameGameEnv instance
        """
        self.agent = agent
        self.env = env

    def train_epochs(
        self,
        epochs: int,
        training_loops: int = 5,
        max_steps: int | None = None,
        report_num: int = 500,
        visualize_num: int = 10,
        update_target_num: int = 1000,
        warmup_episodes: int | None = None,
    ) -> list[float]:
        """Execute training for specified epochs.

        Args:
            epochs: Number of training epochs
            training_loops: Number of learning iterations per episode
            max_steps: Maximum steps per episode
            report_num: Number of report intervals
            visualize_num: Number of visualization intervals
            update_target_num: Number of target update intervals
            warmup_episodes: Number of warmup episodes

        Returns:
            Loss history from training
        """
        return train(
            agent=self.agent,
            env=self.env,
            epochs=epochs,
            training_loops=training_loops,
            max_steps=max_steps,
            report_num=report_num,
            visualize_num=visualize_num,
            update_target_num=update_target_num,
            warmup_episodes=warmup_episodes,
        )


class TrainingManager:
    """Thin coordinator between training orchestrator and checkpoint service.

    Provides high-level training interface with optional checkpoint support.
    Coordinates training execution and checkpoint creation without mixing concerns.
    """

    def __init__(
        self,
        agent,
        env,
        experiment_name: str,
        checkpoint_service=None,
    ):
        """Initialize training manager.

        Args:
            agent: DqnAgent instance
            env: SameGameEnv instance
            experiment_name: Name of the experiment for checkpoint identification
            checkpoint_service: Optional CheckpointService for checkpoint support
        """
        self.agent = agent
        self.env = env
        self.experiment_name = experiment_name
        self.checkpoint_service = checkpoint_service
        self.orchestrator = TrainingOrchestrator(agent, env)

        # Training state tracking
        self.total_steps = 0
        self.cumulative_loss_history = []

    def train(
        self,
        epochs: int,
        training_loops: int = 5,
        max_steps: int | None = None,
        report_num: int = 500,
        visualize_num: int = 10,
        update_target_num: int = 1000,
        warmup_episodes: int | None = None,
    ) -> list[float]:
        """Execute training without checkpointing.

        Args:
            epochs: Number of training epochs
            training_loops: Number of learning iterations per episode
            max_steps: Maximum steps per episode
            report_num: Number of report intervals
            visualize_num: Number of visualization intervals
            update_target_num: Number of target update intervals
            warmup_episodes: Number of warmup episodes

        Returns:
            Loss history from training
        """
        loss_history = self.orchestrator.train_epochs(
            epochs=epochs,
            training_loops=training_loops,
            max_steps=max_steps,
            report_num=report_num,
            visualize_num=visualize_num,
            update_target_num=update_target_num,
            warmup_episodes=warmup_episodes,
        )

        self.cumulative_loss_history.extend(loss_history)
        self.total_steps += epochs

        return loss_history

    def train_with_checkpoints(
        self,
        total_epochs: int,
        checkpoint_every: int,
        random_seed: int = 42,
        training_loops: int = 5,
        max_steps: int | None = None,
        report_num: int = 500,
        visualize_num: int = 10,
        update_target_num: int = 1000,
        warmup_episodes: int | None = None,
        benchmark_results: dict | None = None,
    ) -> list[float]:
        """Execute training with periodic checkpointing.

        Args:
            total_epochs: Total number of training epochs
            checkpoint_every: Create checkpoint every N epochs
            random_seed: Random seed for reproducibility
            training_loops: Number of learning iterations per episode
            max_steps: Maximum steps per episode
            report_num: Number of report intervals
            visualize_num: Number of visualization intervals
            update_target_num: Number of target update intervals
            warmup_episodes: Number of warmup episodes
            benchmark_results: Optional benchmark metrics to include

        Returns:
            Complete loss history from all training

        Raises:
            ValueError: If checkpoint_service is not configured
        """
        if self.checkpoint_service is None:
            raise ValueError(
                "CheckpointService must be provided to use train_with_checkpoints"
            )

        all_loss_history = []
        current_epoch = 0

        while current_epoch < total_epochs:
            # Train for checkpoint_every epochs (or remaining epochs)
            epochs_to_train = min(checkpoint_every, total_epochs - current_epoch)

            loss_history = self.train(
                epochs=epochs_to_train,
                training_loops=training_loops,
                max_steps=max_steps,
                report_num=report_num,
                visualize_num=visualize_num,
                update_target_num=update_target_num,
                warmup_episodes=warmup_episodes if current_epoch == 0 else 0,
            )

            all_loss_history.extend(loss_history)
            current_epoch += epochs_to_train

            # Create checkpoint
            self.checkpoint_service.create_checkpoint(
                agent=self.agent,
                env=self.env,
                experiment_name=self.experiment_name,
                epoch=current_epoch,
                total_epochs=total_epochs,
                total_steps=self.total_steps,
                loss_history=self.cumulative_loss_history[-10:],  # Last 10 loss values
                benchmark_results=benchmark_results,
                random_seed=random_seed,
            )

        return all_loss_history

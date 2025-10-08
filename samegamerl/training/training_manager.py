"""Training manager for coordinating DQN training with optional checkpointing.

Consolidates training loop logic and checkpoint coordination into a single class,
providing a clean interface for both simple training and checkpoint-enabled training.
"""

from tqdm import tqdm


class TrainingManager:
    """Manages DQN training execution with optional checkpoint support.

    Combines training loop logic with checkpoint coordination. Handles warmup,
    epsilon decay, target network updates, and progress reporting.
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

        # Training state tracking
        self.total_steps = 0
        self.cumulative_loss_history = []

    def train(
        self,
        epochs: int,
        training_loops: int = 5,
        max_steps: int = 100,
        report_num: int = 500,
        visualize_num: int = 10,
        update_target_num: int = 1000,
        warmup_episodes: int | None = None,
    ) -> list[float]:
        """Execute training for specified number of epochs.

        Args:
            epochs: Number of training epochs
            training_loops: Number of learning iterations per episode
            max_steps: Maximum steps per episode (defaults to half of total cells)
            report_num: Number of report intervals
            visualize_num: Number of visualization intervals (0 to disable)
            update_target_num: Number of target update intervals
            warmup_episodes: Number of warmup episodes (auto-calculated if None)

        Returns:
            Loss history from training
        """
        if max_steps is None:
            max_steps = self.env.config.total_cells // 2

        report_freq = max(1, epochs // report_num)
        update_target_freq = max(1, epochs // update_target_num)

        if visualize_num == 0:
            visualize_freq = epochs + 1
        else:
            visualize_freq = max(1, epochs // visualize_num)

        # Warmup phase to create sufficient samples in replay buffer
        if warmup_episodes is None:
            if max_steps > 0:
                warmup_episodes = max(1, (self.agent.batch_size // max_steps) * 2)
            else:
                warmup_episodes = 0

        for _ in range(warmup_episodes):
            obs = self.env.reset()

            for step in range(max_steps):
                action = self.agent.act(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.agent.remember(obs, action, reward, next_obs, done)
                obs = next_obs

                if done:
                    break
                if (
                    step > max_steps / 2
                    and self.env.game.get_singles() == self.env.game.left
                ):
                    break

        total_reward = 0
        loss = 0
        results = []
        self.agent.model.train()

        for episode in tqdm(range(epochs)):
            obs = self.env.reset()

            for step in range(max_steps):
                action = self.agent.act(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.agent.remember(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                if done:
                    break

            cur_loss = self.agent.learn()
            loss = (loss + cur_loss) / 2

            for _ in range(training_loops):
                self.agent.learn()

            self.agent.decrease_epsilon()

            if episode % report_freq == report_freq - 1:
                results.append(float(loss))
                loss = 0

            # Commented out visualization - can be enabled if needed
            # if episode % visualize_freq == visualize_freq - 1:
            #     from samegamerl.evaluation.visualize_agent import play_eval_game
            #     play_eval_game(self.agent, visualize=True, waiting_time=1000)

            if episode % update_target_freq == 0:
                self.agent.update_target_model()

        self.cumulative_loss_history.extend(results)
        self.total_steps += epochs

        return results

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

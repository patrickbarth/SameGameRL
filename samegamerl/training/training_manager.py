"""Training manager for coordinating DQN training with optional checkpointing.

Provides composable primitives for training orchestration: warmup(), train(),
create_checkpoint(), and rollback_to_checkpoint(). Experiments compose these
methods to implement custom training strategies with adaptive checkpointing.
"""

import torch
from tqdm import tqdm

from samegamerl.training.checkpoint_state_extractor import CheckpointStateExtractor


class TrainingManager:
    """Manages DQN training execution with composable checkpoint operations.

    Provides explicit methods for each training operation (warmup, train,
    checkpoint, rollback) allowing experiments to implement adaptive training
    strategies. Tracks cumulative training state across multiple train() calls.
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
        self.extractor = CheckpointStateExtractor()

        # Training state tracking
        self.current_epoch = 0
        self.total_steps = 0
        self.cumulative_loss_history = []

    def warmup(self, episodes: int, max_steps: int | None = None):
        """Populate replay buffer with random play experiences.

        Runs episodes using epsilon-greedy exploration to fill the replay buffer
        before training begins. Can be called multiple times - replay buffer
        manages its own capacity.

        Args:
            episodes: Number of warmup episodes to run
            max_steps: Maximum steps per episode (defaults to half of total cells)
        """
        steps_per_episode = (
            max_steps if max_steps is not None else self.env.config.total_cells // 2
        )

        for _ in range(episodes):
            obs = self.env.reset()

            for step in range(steps_per_episode):
                action = self.agent.act(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.agent.remember(obs, action, reward, next_obs, done)
                obs = next_obs

                if done:
                    break
                if step > steps_per_episode / 2:
                    break

    def train(
        self,
        epochs: int,
        training_loops: int = 5,
        max_steps: int = 100,
        report_num: int = 500,
        visualize_num: int = 10,
        update_target_num: int = 1000,
    ) -> list[float]:
        """Execute training for specified number of epochs.

        Args:
            epochs: Number of training epochs
            training_loops: Number of learning iterations per episode
            max_steps: Maximum steps per episode (defaults to half of total cells)
            report_num: Number of report intervals
            visualize_num: Number of visualization intervals (0 to disable)
            update_target_num: Number of target update intervals

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
        self.current_epoch += epochs

        return results

    def create_checkpoint(self) -> str:
        """Create a checkpoint at the current training state.

        Saves current model weights, optimizer state, epsilon, and environment
        configuration. Checkpoint ID is auto-generated from experiment name
        and current epoch.

        Returns:
            Checkpoint identifier string (format: {experiment_name}_epoch_{epoch})

        Raises:
            ValueError: If checkpoint_service is not configured
        """
        if self.checkpoint_service is None:
            raise ValueError(
                "CheckpointService must be provided to use create_checkpoint"
            )

        checkpoint_id = self.checkpoint_service.create_checkpoint(
            agent=self.agent,
            env=self.env,
            experiment_name=self.experiment_name,
            epoch=self.current_epoch,
            total_epochs=self.current_epoch,
            total_steps=self.total_steps,
            loss_history=self.cumulative_loss_history[-10:],
            benchmark_results=None,
            random_seed=42,
        )

        return checkpoint_id

    def rollback_to_checkpoint(self, checkpoint_id: str):
        """Restore complete training state from checkpoint.

        Loads checkpoint and restores all agent and environment state including
        model weights, target model, optimizer state, epsilon, and training
        progress. Clears the replay buffer - experiment should call warmup()
        after rollback to re-populate buffer from the restored model state.

        Args:
            checkpoint_id: Checkpoint identifier to restore from

        Raises:
            ValueError: If checkpoint_service is not configured
            FileNotFoundError: If checkpoint doesn't exist
        """
        if self.checkpoint_service is None:
            raise ValueError(
                "CheckpointService must be provided to use rollback_to_checkpoint"
            )

        # Load checkpoint data
        checkpoint = self.checkpoint_service.load_checkpoint(checkpoint_id)

        # Restore agent hyperparameters (epsilon, learning_rate, etc.)
        self.extractor.restore_agent_state(checkpoint.agent_state, self.agent)

        # Load model weights (model, target_model, optimizer)
        model_path = self.checkpoint_service.repository.get_model_path(checkpoint_id)
        checkpoint_dict = torch.load(model_path, map_location=self.agent.device)
        self.agent.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.agent.target_model.load_state_dict(
            checkpoint_dict["target_model_state_dict"]
        )
        self.agent.opt.load_state_dict(checkpoint_dict["optimizer_state_dict"])

        # Clear replay buffer (experiment will warmup after rollback)
        self.extractor.clear_replay_buffer(self.agent)

        # Reset training state to checkpoint values
        self.current_epoch = checkpoint.epoch
        self.total_steps = checkpoint.training_state.total_steps
        # Note: cumulative_loss_history is NOT restored - it's experiment's history

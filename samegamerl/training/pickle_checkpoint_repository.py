"""Pickle-based checkpoint repository for file storage.

Provides simple file-based checkpoint persistence using Python's pickle module.
This is the primary storage backend, especially important for remote GPU machines
where database dependencies may not be available.
"""

import pickle
from pathlib import Path

from samegamerl.training.checkpoint_data import CheckpointData


class PickleCheckpointRepository:
    """Repository for storing and retrieving checkpoints as pickle files.

    Each checkpoint is stored as a separate .pkl file named by its identifier
    (format: {experiment_name}_epoch_{epoch}.pkl). Model weights are stored
    alongside checkpoint metadata in the same directory.

    Directory structure:
        checkpoint_dir/
            experiment_1_epoch_100.pkl      # Checkpoint metadata
            experiment_1_epoch_100_model.pth  # Model weights
            experiment_1_epoch_200.pkl
            experiment_1_epoch_200_model.pth
            ...

    This storage backend is lightweight, requires no external dependencies,
    and works reliably on remote GPU machines.
    """

    def __init__(self, checkpoint_dir: Path | str):
        """Initialize repository with checkpoint directory.

        Creates the directory if it doesn't exist. All checkpoint files
        (metadata and model weights) are stored in this single directory.

        Args:
            checkpoint_dir: Directory path for storing all checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, checkpoint_id: str) -> Path:
        """Get the path where model weights should be stored.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Path for model weights file
        """
        return self.checkpoint_dir / f"{checkpoint_id}_model.pth"

    def save(self, checkpoint: CheckpointData) -> str:
        """Save checkpoint to pickle file.

        Overwrites existing checkpoint if one exists with same ID.

        Args:
            checkpoint: CheckpointData to save

        Returns:
            Checkpoint identifier string
        """
        checkpoint_id = checkpoint.get_identifier()
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"

        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)

        return checkpoint_id

    def load(self, checkpoint_id: str) -> CheckpointData:
        """Load checkpoint from pickle file.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            CheckpointData loaded from file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if checkpoint file exists
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        return checkpoint_file.exists()

    def delete(self, checkpoint_id: str) -> None:
        """Delete checkpoint file and associated model weights.

        Args:
            checkpoint_id: Checkpoint identifier

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        checkpoint_file.unlink()

        # Also delete model weights if they exist
        model_file = self.get_model_path(checkpoint_id)
        if model_file.exists():
            model_file.unlink()

    def list_checkpoints(self, experiment_name: str | None = None) -> list[str]:
        """List all checkpoint identifiers.

        Args:
            experiment_name: If provided, filter to checkpoints for this experiment

        Returns:
            List of checkpoint identifiers
        """
        checkpoint_files = self.checkpoint_dir.glob("*.pkl")
        checkpoint_ids = [f.stem for f in checkpoint_files]

        if experiment_name:
            # Filter to checkpoints matching experiment name
            checkpoint_ids = [
                cid for cid in checkpoint_ids if cid.startswith(f"{experiment_name}_epoch_")
            ]

        return sorted(checkpoint_ids)

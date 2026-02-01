import orbax.checkpoint
import os
import shutil
import jax

class CheckpointManager:
    """
    Manages saving/loading of PyTree checkpoints using Orbax.
    Supports async saving for minimal training overlap.
    """
    def __init__(self, directory, max_keep=3):
        self.directory = os.path.abspath(directory)
        
        # Orbax Options
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_keep, create=True)
        
        # Checkpointer (Standard PyTree)
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
        # Manager
        self.manager = orbax.checkpoint.CheckpointManager(
            self.directory, self.checkpointer, options
        )
        print(f"[CheckpointManager] Initialized at {self.directory}")

    def save(self, step, items):
        """
        Args:
            step: int step number
            items: dict of {'model': model, 'opt_state': ...} or tuple
        """
        save_args = orbax.checkpoint.args.PyTreeSave(item=items)
        self.manager.save(step, args=save_args)
        # Wait until save is finished? For async we assume it handles it.
        # But for safety in this script we might block on completion if needed.
        # self.manager.wait_until_finished() 
        print(f"   >> 💾 Orbax Snapshot queued: step_{step}")

    def restore(self, step, item_structure):
        """
        Restores items. Must provide item_structure (mock layout) to guide restoration.
        """
        restore_args = orbax.checkpoint.args.PyTreeRestore(item=item_structure)
        restored = self.manager.restore(step, args=restore_args)
        return restored

    def latest_step(self):
        return self.manager.latest_step()

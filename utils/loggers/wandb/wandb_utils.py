# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# WARNING ⚠️ wandb is deprecated and will be removed in future release.
# See supported integrations at https://github.com/ultralytics/yolov5#integrations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from utils.general import LOGGER, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
RANK = int(os.getenv("RANK", -1))
DEPRECATION_WARNING = (
    f"{colorstr('wandb')}: WARNING ⚠️ wandb is deprecated and will be removed in a future release. "
    f"See supported integrations at https://github.com/ultralytics/yolov5#integrations."
)

try:
    import wandb
    assert hasattr(wandb, "__version__")  # verify package import not local dir
    LOGGER.warning(DEPRECATION_WARNING)
except (ImportError, AssertionError):
    wandb = None


class WandbLogger:
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information includes hyperparameters, system
    configuration and metrics, model metrics, and basic data metrics and analyses.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_id=None, job_type="Training"):
        """
        - Initialize WandbLogger instance
        - Upload dataset if opt.upload_dataset is True (omitted here unless you add artifact helpers)
        - Setup training processes if job_type is 'Training'.

        Args:
            opt (namespace): Command-line arguments / options.
            run_id (str | None): W&B run id for resume.
            job_type (str): W&B job type label.
        """
        self.job_type = job_type
        self.wandb = wandb
        self.wandb_run = wandb.run if wandb else None

        # Artifacts / tables
        self.val_artifact, self.train_artifact = None, None
        self.train_artifact_path, self.val_artifact_path = None, None
        self.result_artifact = None
        self.val_table, self.result_table = None, None
        self.max_imgs_to_log = 16
        self.data_dict = None

        # ---- SAFE GUARDS: allow disabling and missing args without crashing ----
        disabled_flag = getattr(opt, "disable_wandb", False)
        env_disabled = os.environ.get("WANDB_DISABLED", "").lower() == "true"
        if disabled_flag or env_disabled or self.wandb is None:
            # Do not initialize wandb; training continues without logging
            self.wandb_run = None
        else:
            # Resolve safe defaults
            try:
                # project: if "runs/train" keep "YOLOv3" for backward compat, else use last path component
                wb_project = getattr(opt, "wb_project", None)
               
                if wb_project:  # set via --wb-project
                    project = wb_project
                else: # keep prior fallback so it still works without --wb-project
                    if getattr(opt, "project", None):
                        project = "YOLOv3" if str(opt.project) == "runs/train" else Path(opt.project).stem
                    else:
                        project = "YOLOv3"

                entity = getattr(opt, "entity", None)  # None -> personal account or env default
                name = getattr(opt, "name", None)
                if not name or name == "exp":
                    name = None  # let wandb auto-name

                # Use a dict for config to avoid serialization issues
                try:
                    cfg = {k: v for k, v in vars(opt).items()}
                except Exception:
                    cfg = {}

                # Initialize run (safe)
                self.wandb_run = self.wandb.run or self.wandb.init(
                    config=cfg,
                    resume="allow",
                    project=project,
                    entity=entity,
                    name=name,
                    job_type=job_type,
                    id=run_id,
                    allow_val_change=True,
                )
            except Exception as e:
                # If wandb init fails, continue training without wandb
                print(f"[W&B] init failed: {e}", file=sys.stderr)
                self.wandb_run = None

        # Training-only setup
        if self.wandb_run and self.job_type == "Training":
            if isinstance(opt.data, dict):
                # Another integration may have preprocessed the dataset dict
                self.data_dict = opt.data
            self.setup_training(opt)

    def setup_training(self, opt):
        """
        Setup processes for training:
          - (Optional) artifact resume/download logic can be added here.
          - Initialize logging dict and bbox log interval.
        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval

        if isinstance(opt.resume, str):
            # NOTE: original YOLOv5 code may download artifacts here.
            # Add your artifact helpers if you actually use W&B artifacts.
            pass

        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # disable bbox logging

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as a W&B artifact.

        Args:
            path (Path): Directory containing checkpoints.
            opt (namespace): CLI options.
            epoch (int): Current epoch index.
            fitness_score (float): Fitness score.
            best_model (bool): Whether this is the best checkpoint so far.
        """
        if self.wandb_run and self.wandb:
            try:
                model_artifact = self.wandb.Artifact(
                    f"run_{self.wandb.run.id}_model",
                    type="model",
                    metadata={
                        "original_url": str(path),
                        "epochs_trained": epoch + 1,
                        "save period": opt.save_period,
                        "project": getattr(opt, "project", None),
                        "total_epochs": opt.epochs,
                        "fitness_score": fitness_score,
                    },
                )
                model_artifact.add_file(str(path / "last.pt"), name="last.pt")
                self.wandb.log_artifact(
                    model_artifact,
                    aliases=[
                        "latest",
                        "last",
                        f"epoch {str(self.current_epoch)}",
                        "best" if best_model else "",
                    ],
                )
                LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")
            except Exception as e:
                print(f"[W&B] log_model failed: {e}", file=sys.stderr)

    def val_one_image(self, pred, predn, path, names, im):
        """Evaluates model prediction for a single image (placeholder for compatibility)."""
        pass

    def log(self, log_dict):
        """
        Stage metrics to internal dict; flushed on end_epoch().

        Args:
            log_dict (dict): metrics/media to log this step.
        """
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self):
        """
        Commit staged logs to W&B and clear the staging dict.
        """
        if self.wandb_run and self.wandb:
            with all_logging_disabled():
                try:
                    self.wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(
                        f"An error occurred in wandb. Training will proceed without interruption.\n{e}"
                    )
                    # gracefully disable wandb if it errors mid-run
                    try:
                        self.wandb_run.finish()
                    except Exception:
                        pass
                    self.wandb_run = None
                self.log_dict = {}

    def finish_run(self):
        """
        Flush remaining logs (if any) and finish the W&B run.
        """
        if self.wandb_run and self.wandb:
            if self.log_dict:
                with all_logging_disabled():
                    try:
                        self.wandb.log(self.log_dict)
                    except Exception:
                        pass
            try:
                self.wandb.run.finish()
            except Exception:
                pass
            LOGGER.warning(DEPRECATION_WARNING)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages triggered during the body from being processed.

    Args:
        highest_level (int): maximum logging level in use.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)

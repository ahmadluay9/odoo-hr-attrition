"""MLflow Model Registry and Champion-Challenger governance manager."""

from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.config.settings import get_settings

settings = get_settings()


class ModelRegistryManager:
    """Manages model versioning, champion/challenger promotions, and model loading."""

    def __init__(self):
        self.tracking_uri = settings.mlflow_tracking_uri
        self.model_name = settings.mlflow_model_name
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_and_govern(
        self,
        model_uri: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        force_champion: bool = False,
    ) -> dict[str, Any]:
        """Register a model run, compare against champion, and promote if superior.

        Returns metadata dictionary including version, promoted status, and comparison.
        """
        # Register new version
        mv = mlflow.register_model(model_uri=model_uri, name=self.model_name)
        version = mv.version

        # Tag version with performance metrics
        for k, v in metrics.items():
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key=k,
                value=str(round(v, 4)),
            )

        new_pr_auc = metrics.get("pr_auc", 0.0)

        # Check existing champion
        champion_mv = None
        try:
            champion_mv = self.client.get_model_version_by_alias(
                name=self.model_name, alias=settings.champion_alias
            )
        except (RestException, Exception):
            pass

        promoted = False
        reason = ""

        if force_champion or champion_mv is None:
            # First model or forced promotion
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=settings.champion_alias,
                version=version,
            )
            promoted = True
            reason = (
                "Initial champion assigned."
                if champion_mv is None
                else "Forced champion promotion."
            )
        else:
            # Compare challenger against current champion
            champ_pr_auc = float(champion_mv.tags.get("pr_auc", 0.0))
            if new_pr_auc > champ_pr_auc:
                # Promote to champion
                self.client.set_registered_model_alias(
                    name=self.model_name,
                    alias=settings.champion_alias,
                    version=version,
                )
                # Mark previous champion
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=champion_mv.version,
                    key="previous_status",
                    value="superseded_champion",
                )
                promoted = True
                reason = f"Challenger PR-AUC ({new_pr_auc:.4f}) exceeded Champion PR-AUC ({champ_pr_auc:.4f})."
            else:
                # Set as challenger
                self.client.set_registered_model_alias(
                    name=self.model_name,
                    alias=settings.challenger_alias,
                    version=version,
                )
                promoted = False
                reason = f"Challenger PR-AUC ({new_pr_auc:.4f}) did not beat Champion PR-AUC ({champ_pr_auc:.4f})."

        print(
            f"[Model Registry] Version {version} registered. Promoted: {promoted}. Details: {reason}"
        )
        return {
            "version": version,
            "promoted": promoted,
            "reason": reason,
            "current_champion_version": version
            if promoted
            else (champion_mv.version if champion_mv else None),
            "pr_auc": new_pr_auc,
        }

    def load_champion_model(self):
        """Load the active Champion model from MLflow registry."""
        model_uri = f"models:/{self.model_name}@{settings.champion_alias}"
        return mlflow.pyfunc.load_model(model_uri)

    def get_champion_version(self):
        """Get the model version metadata for active champion."""
        try:
            return self.client.get_model_version_by_alias(
                name=self.model_name, alias=settings.champion_alias
            )
        except Exception:
            return None

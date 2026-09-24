"""Module obsolète : la phase DISCOVERY est désormais déterministe.

Historiquement, ce module pilotait un agent LLM de lecture seule pour explorer
le workspace. La conception retenue déplace la découverte dans un code Python
entièrement déterministe intégré à l'orchestrateur (voir
``core.multi_agent.orchestrator.ChefDeProjet.discover`` et ``DiscoveryReport``).

Ce fichier est conservé uniquement pour compatibilité d'import. Il ne contient
plus aucune logique active. Tout nouveau code doit utiliser l'orchestrateur.
"""
from __future__ import annotations

from core.multi_agent.orchestrator import DiscoveryReport

__all__ = ["DiscoveryReport"]

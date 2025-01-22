# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module consists of custom exceptions for end to end testing"""

class PersistentStoreCreationException(Exception):
    """Exception for persistent store creation"""
    pass


class DockerException(Exception):
    """Exception for docker"""
    pass


class PlanModificationException(Exception):
    """Exception for plan modification"""
    pass


class WorkspaceCertificationException(Exception):
    """Exception for workspace certification"""
    pass


class CollaboratorRegistrationException(Exception):
    """Exception for collaborator registration"""
    pass


class PlanInitializationException(Exception):
    """Exception for plan initialization"""
    pass


class CSRGenerationException(Exception):
    """Exception for cert sign request generation"""
    pass


class AggregatorCertificationException(Exception):
    """Exception for aggregator certification"""
    pass


class WorkspaceExportException(Exception):
    """Exception for workspace export"""
    pass


class WorkspaceImportException(Exception):
    """Exception for workspace import"""
    pass


class CollaboratorCreationException(Exception):
    """Exception for aggregator creation"""
    pass


class WorkspaceDockerizationException(Exception):
    """Exception for workspace dockerization"""
    pass


class WorkspaceLoadException(Exception):
    """Exception for workspace load"""
    pass


class ReferenceFlowException(Exception):
    """Exception for reference flow"""
    pass


class NotebookRunException(Exception):
    """Exception for notebook run"""
    pass


class EnvoyStartException(Exception):
    """Exception for envoy start"""
    pass


class DirectorStartException(Exception):
    """Exception for director start"""
    pass


class DataSetupException(Exception):
    """Exception for data setup for given model"""
    pass

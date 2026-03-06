"""Standardized service-layer exceptions for controller mapping."""

from __future__ import annotations


class ServiceError(Exception):
    status_code = 500


class InvalidRequestError(ServiceError):
    status_code = 400


class NotFoundError(ServiceError):
    status_code = 404


class DependencyUnavailableError(ServiceError):
    status_code = 501

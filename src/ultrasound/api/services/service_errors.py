"""Standardized service-layer exceptions for controller mapping."""

from __future__ import annotations


class ServiceError(Exception):
    status_code = 500


class InvalidRequestError(ServiceError):
    status_code = 400


class UnauthorizedError(ServiceError):
    status_code = 401


class ForbiddenError(ServiceError):
    status_code = 403


class NotFoundError(ServiceError):
    status_code = 404


class DependencyUnavailableError(ServiceError):
    status_code = 501

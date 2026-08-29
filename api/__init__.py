"""Thin HTTP API exposing the email agent as a service.

The API layer contains no business logic: it serializes requests/responses and
delegates to the EmailAgent orchestrator. Everything the agent knows how to do
lives in the ``agent`` package.
"""

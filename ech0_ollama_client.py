#!/usr/bin/env python3
"""Thin client for interacting with the local Ollama runtime."""

import os
import time
from typing import List, Dict, Optional

import requests


class OllamaClient:
    """Utility wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        model: str = None,
        host: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout: int = 120,
        keep_alive: str = "5m",
    ):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("ECH0_OLLAMA_MODEL") or "ech0-14b-v4:latest"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.keep_alive = keep_alive

    def ensure_model(self):
        """Verify the requested model is available locally."""
        response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
        available = {entry.get("model") or entry.get("name") for entry in models}
        if self.model not in available:
            raise RuntimeError(
                f"Model '{self.model}' not found in Ollama. Run 'ollama pull {self.model}' and retry."
            )

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send a prompt to the Ollama chat API and return the response text."""
        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

        conversation = messages[:] if messages else []
        if system:
            conversation.insert(0, {"role": "system", "content": system})
        conversation.append({"role": "user", "content": prompt})
        payload["messages"] = conversation

        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    def embed(self, text: str) -> List[float]:
        """Generate embeddings via Ollama (best-effort; not all models support it)."""
        payload = {"model": self.model, "input": text}
        response = requests.post(
            f"{self.host}/api/embeddings",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

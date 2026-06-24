import requests
import json
import google.generativeai as genai
from typing import Dict, Any, List
from backend.config import settings

class LLMGenerator:
    def __init__(self):
        self.gemini_configured = False
        self._setup_gemini()

    def _setup_gemini(self):
        """Sets up the Google Gemini client if the API key is present."""
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_configured = True
                print("Gemini API configured successfully.")
            except Exception as e:
                print(f"Failed to configure Gemini API: {str(e)}")
        else:
            print("Gemini API Key not found. Gemini generation will be unavailable until a key is provided.")

    def update_gemini_key(self, api_key: str):
        """Allows updating the Gemini API key dynamically from the UI."""
        if api_key:
            settings.GEMINI_API_KEY = api_key
            self._setup_gemini()

    def generate_response(self, query: str, chunks: List[Dict[str, Any]], provider: str = "simulation", model: str = "gemini-1.5-flash") -> Dict[str, Any]:
        """Generates an answer using the retrieved context, query, and selected LLM provider."""
        # 1. Build context text
        context_text = ""
        for idx, chunk in enumerate(chunks):
            context_text += f"Source [{idx + 1}] (File: {chunk['file_name']}):\n{chunk['text']}\n\n"

        # 2. Build system-instruction-focused prompt
        system_prompt = (
            "You are a helpful, precise documentation assistant. Use ONLY the provided context sources below to answer the query. "
            "If the answer cannot be found in the context, politely state that you cannot find the answer in the provided documents.\n\n"
            f"Context sources:\n{context_text}\n"
            f"User Query: {query}\n\n"
            "Provide a comprehensive, accurate answer. Always cite your sources by referencing their Source number (e.g. [1], [2]) when asserting facts."
        )

        # 3. Dispatch to selected provider
        if provider == "gemini":
            return self._generate_gemini(system_prompt, model)
        elif provider == "ollama":
            return self._generate_ollama(system_prompt, model)
        else:
            return self._generate_simulation(query, chunks)

    def _generate_gemini(self, prompt: str, model_name: str) -> Dict[str, Any]:
        if not self.gemini_configured:
            return {
                "answer": "Error: Gemini API key is not set. Please set it in the configuration or enter it in the dashboard UI.",
                "provider": "gemini",
                "model": model_name,
                "error": True
            }
        try:
            # Use gemini-1.5-flash as default
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "provider": "gemini",
                "model": model_name,
                "error": False
            }
        except Exception as e:
            return {
                "answer": f"Gemini Generation failed: {str(e)}",
                "provider": "gemini",
                "model": model_name,
                "error": True
            }

    def _generate_ollama(self, prompt: str, model_name: str) -> Dict[str, Any]:
        url = f"{settings.OLLAMA_API_URL}/api/generate"
        payload = {
            "model": model_name or settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return {
                    "answer": result.get("response", ""),
                    "provider": "ollama",
                    "model": model_name,
                    "error": False
                }
            else:
                return {
                    "answer": f"Ollama returned status code: {response.status_code}. Response: {response.text}",
                    "provider": "ollama",
                    "model": model_name,
                    "error": True
                }
        except requests.exceptions.ConnectionError:
            return {
                "answer": f"Connection Error: Could not connect to local Ollama instance at {settings.OLLAMA_API_URL}. Ensure Ollama is running (`ollama serve`) and the model '{model_name}' is downloaded (`ollama pull {model_name}`).",
                "provider": "ollama",
                "model": model_name,
                "error": True
            }
        except Exception as e:
            return {
                "answer": f"Ollama Generation failed: {str(e)}",
                "provider": "ollama",
                "model": model_name,
                "error": True
            }

    def _generate_simulation(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provides a mock response showing what the system received, perfect for running without any LLM keys."""
        if not chunks:
            return {
                "answer": "*(Simulation Mode)*\nNo context was retrieved, so I cannot answer your query. Try uploading a document and index it first!",
                "provider": "simulation",
                "model": "local-sim",
                "error": False
            }
            
        sources_list = ", ".join([f"**Source [{c['global_id'] + 1}]** (Score: {c.get('score', 0):.4f})" for c in chunks])
        
        sim_response = (
            "### 🤖 RAG Simulation Mode Answer\n\n"
            "This is a simulated LLM response because no external API key (Gemini) or local LLM server (Ollama) was configured. "
            "However, the RAG retrieval pipeline worked perfectly!\n\n"
            "Here is the context retrieved from your document database:\n"
        )
        
        for idx, chunk in enumerate(chunks):
            snippet = chunk['text'][:150].replace('\n', ' ') + "..."
            sim_response += f"- **[{idx + 1}]** `{chunk['file_name']}`: \"{snippet}\"\n"
            
        sim_response += (
            f"\n**Synthesizing Answer to Query**: *\"{query}\"*\n\n"
            f"Based on the {len(chunks)} sources retrieved (specifically {sources_list}), "
            "the system is ready to compile an answer. To generate a real text response, please: \n"
            "1. Enter a **Google Gemini API Key** in the settings panel (completely free and no credit card required).\n"
            "2. Or, run **Ollama** locally on your machine."
        )
        
        return {
            "answer": sim_response,
            "provider": "simulation",
            "model": "local-sim",
            "error": False
        }

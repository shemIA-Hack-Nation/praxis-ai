"""
LLM Provider abstraction layer for supporting multiple LLM providers.
"""
import logging
from typing import Optional, List, Dict, Any
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """Hugging Face Inference API LLM wrapper for LangChain-like interface.
    
    Supports models via Hugging Face Inference API, including:
    - meta-llama/Meta-Llama-3-70B
    - Other models available on Hugging Face
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-70B", api_key: Optional[str] = None, temperature: float = 0.7):
        """Initialize Hugging Face LLM.
        
        Args:
            model_name: Hugging Face model name (default: "meta-llama/Meta-Llama-3-70B")
            api_key: Hugging Face API token (optional, will use config or env)
            temperature: Temperature for generation (0.0-1.0)
        """
        try:
            import os
            from huggingface_hub import InferenceClient
            
            # Use provided key, then env, then config (with fallback token)
            self.api_key = api_key or os.environ.get("HF_TOKEN") or settings.HF_TOKEN or "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"
            self.model_name = model_name
            self.temperature = temperature
            
            if self.api_key:
                self.client = InferenceClient(
                    provider="auto",
                    api_key=self.api_key,
                )
                logger.info(f"Hugging Face LLM initialized with model: {model_name}")
            else:
                self.client = None
                logger.warning("Hugging Face API token not provided. Set HF_TOKEN environment variable.")
        except ImportError:
            logger.error("huggingface_hub not installed. Install it with: pip install huggingface_hub")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Hugging Face LLM: {str(e)}")
            self.client = None
    
    async def ainvoke(self, messages: List[Any]) -> Any:
        """Async invoke method compatible with LangChain."""
        if not self.client:
            raise ValueError("Hugging Face client not initialized")
        
        # Convert LangChain messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Run in executor since InferenceClient is sync
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.text_generation(
                    prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_new_tokens=1024,  # Reduced from 2048 to save tokens
                )
            )
            
            # Extract just the assistant's response (not the full conversation)
            # The response might include the full conversation, extract just the last part
            if isinstance(response, str):
                # If response contains conversation history, extract just the assistant's response
                if "Assistant:" in response:
                    # Extract everything after the last "Assistant:"
                    parts = response.split("Assistant:")
                    if len(parts) > 1:
                        response = parts[-1].strip()
                elif "User:" in response:
                    # If it's a conversation, extract the last assistant response
                    # Look for JSON or structured content
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        response = json_match.group(0)
            
            # Return LangChain-like response
            return type('Response', (), {
                'content': response
            })()
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            raise
    
    def invoke(self, messages: List[Any]) -> Any:
        """Sync invoke method."""
        if not self.client:
            raise ValueError("Hugging Face client not initialized")
        
        prompt = self._messages_to_prompt(messages)
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_new_tokens=2048,
            )
            
            return type('Response', (), {
                'content': response
            })()
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            raise
    
    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert LangChain messages to prompt format."""
        prompt_parts = []
        system_instruction = None
        
        for message in messages:
            # Check if it's a LangChain message object
            if hasattr(message, 'content'):
                content = message.content
                # Check message type
                message_type = None
                if hasattr(message, '__class__'):
                    class_name = message.__class__.__name__
                    if 'SystemMessage' in class_name or 'system' in class_name.lower():
                        message_type = 'system'
                    elif 'HumanMessage' in class_name or 'human' in class_name.lower():
                        message_type = 'human'
                    elif 'AIMessage' in class_name or 'ai' in class_name.lower():
                        message_type = 'ai'
                
                # Also check type attribute if available
                if not message_type and hasattr(message, 'type'):
                    message_type = message.type
                
                if message_type == 'system':
                    system_instruction = content
                elif message_type == 'human':
                    prompt_parts.append(content)
                elif message_type == 'ai':
                    # Include AI messages in conversation format
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    # Assume it's a human message
                    prompt_parts.append(content)
            else:
                # If it's already a string
                prompt_parts.append(str(message))
        
        # Combine system instruction with prompt
        # For Llama models, format as: System: ... User: ... Assistant: ...
        full_prompt = ""
        if system_instruction:
            full_prompt += f"System: {system_instruction}\n\n"
        
        # Add user messages
        user_content = "\n\n".join(prompt_parts)
        full_prompt += f"User: {user_content}\n\nAssistant:"
        
        return full_prompt


class GeminiLLM:
    """Gemini LLM wrapper for LangChain-like interface.
    
    Supports free tier models: gemini-pro, gemini-1.5-flash
    Paid tier models: gemini-1.5-pro, gemini-pro-vision
    """
    
    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None, temperature: float = 0.7):
        """Initialize Gemini LLM.
        
        Args:
            model_name: Gemini model name. Free tier: "gemini-pro" (default) or "gemini-1.5-flash"
            api_key: Gemini API key (optional, will use config or fallback)
            temperature: Temperature for generation (0.0-1.0)
        """
        try:
            import google.generativeai as genai
            self.genai = genai
            # Use provided key, then env, then default to the hardcoded key as fallback
            self.api_key = api_key or settings.GEMINI_API_KEY or "AIzaSyA9a3OHjxLiTO07EmUbqVEw3GbA0xZRMYo"
            self.model_name = model_name
            self.temperature = temperature
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"Gemini LLM initialized with model: {model_name}")
            else:
                self.model = None
                logger.warning("Gemini API key not provided")
        except ImportError:
            logger.error("google-generativeai not installed. Install it with: pip install google-generativeai")
            self.model = None
            self.genai = None
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {str(e)}")
            self.model = None
            self.genai = None
    
    async def ainvoke(self, messages: List[Any]) -> Any:
        """Async invoke method compatible with LangChain."""
        if not self.model:
            raise ValueError("Gemini model not initialized")
        
        # Convert LangChain messages to Gemini format
        prompt = self._messages_to_prompt(messages)
        
        # Run in executor since Gemini SDK is sync
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=self.temperature
                    )
                )
            )
            
            # Extract text from response
            text = ""
            if hasattr(response, 'text') and response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to get text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text += part.text
            
            # Return LangChain-like response
            return type('Response', (), {
                'content': text
            })()
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def invoke(self, messages: List[Any]) -> Any:
        """Sync invoke method."""
        if not self.model:
            raise ValueError("Gemini model not initialized")
        
        prompt = self._messages_to_prompt(messages)
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )
            
            # Extract text from response
            text = ""
            if hasattr(response, 'text') and response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to get text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text += part.text
            
            return type('Response', (), {
                'content': text
            })()
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert LangChain messages to Gemini prompt format."""
        prompt_parts = []
        system_instruction = None
        
        for message in messages:
            # Check if it's a LangChain message object
            if hasattr(message, 'content'):
                content = message.content
                # Check message type - LangChain uses __class__.__name__ or type attribute
                message_type = None
                if hasattr(message, '__class__'):
                    class_name = message.__class__.__name__
                    if 'SystemMessage' in class_name or 'system' in class_name.lower():
                        message_type = 'system'
                    elif 'HumanMessage' in class_name or 'human' in class_name.lower():
                        message_type = 'human'
                    elif 'AIMessage' in class_name or 'ai' in class_name.lower():
                        message_type = 'ai'
                
                # Also check type attribute if available
                if not message_type and hasattr(message, 'type'):
                    message_type = message.type
                
                if message_type == 'system':
                    system_instruction = content
                elif message_type == 'human':
                    prompt_parts.append(content)
                elif message_type == 'ai':
                    # Skip AI messages in prompt construction for Gemini
                    pass
                else:
                    # Assume it's a human message
                    prompt_parts.append(content)
            else:
                # If it's already a string
                prompt_parts.append(str(message))
        
        # Combine system instruction with prompt
        full_prompt = "\n\n".join(prompt_parts)
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{full_prompt}"
        
        return full_prompt


class GroqLLM:
    """Groq LLM wrapper for LangChain-like interface.
    
    Groq provides fast inference for models like:
    - llama-3.1-8b-instant (fast, recommended)
    - llama-3.1-70b-versatile (if available)
    - mixtral-8x7b-32768
    - gemma2-9b-it
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: Optional[str] = None, temperature: float = 0.7):
        """Initialize Groq LLM.
        
        Args:
            model_name: Groq model name (default: "llama-3.1-8b-instant")
            api_key: Groq API key (optional, will use config or env)
            temperature: Temperature for generation (0.0-1.0)
        """
        try:
            import os
            from groq import Groq
            
            # Use provided key, then env, then config
            self.api_key = api_key or os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
            self.model_name = model_name
            self.temperature = temperature
            
            if self.api_key:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq LLM initialized with model: {model_name}")
            else:
                self.client = None
                logger.warning("Groq API key not provided. Set GROQ_API_KEY environment variable.")
        except ImportError:
            logger.error("groq not installed. Install it with: pip install groq")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Groq: {str(e)}")
            self.client = None
    
    async def ainvoke(self, messages: List[Any]) -> Any:
        """Async invoke method compatible with LangChain with rate limiting and retry logic."""
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        # Convert LangChain messages to Groq format
        groq_messages = self._messages_to_groq_format(messages)
        
        # Retry logic with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # Run in executor since Groq SDK might be sync
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model_name,
                        messages=groq_messages,
                        temperature=self.temperature,
                        max_tokens=1024,  # Reduced from 2048 to save tokens
                    )
                )
                
                # Extract content from response
                content = ""
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content or ""
                
                # Extract just the assistant's response if it contains conversation
                if isinstance(content, str):
                    if "Assistant:" in content:
                        parts = content.split("Assistant:")
                        if len(parts) > 1:
                            content = parts[-1].strip()
                    elif "User:" in content:
                        # Extract JSON or structured content
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            content = json_match.group(0)
                
                # Return LangChain-like response
                return type('Response', (), {
                    'content': content
                })()
                
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str or "tokens per minute" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        import re
                        wait_match = re.search(r'try again in ([\d.]+)s', error_str, re.IGNORECASE)
                        if wait_match:
                            wait_time = float(wait_match.group(1)) + 0.5  # Add buffer
                        else:
                            wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                        
                        logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries: {str(e)}")
                        raise
                else:
                    # Not a rate limit error, raise immediately
                    logger.error(f"Error calling Groq API: {str(e)}")
                    raise
        
        # Should not reach here, but just in case
        raise Exception("Failed to get response after retries")
    
    def invoke(self, messages: List[Any]) -> Any:
        """Sync invoke method."""
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        groq_messages = self._messages_to_groq_format(messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=2048,
            )
            
            content = ""
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""
            
            # Extract just the assistant's response if needed
            if isinstance(content, str):
                if "Assistant:" in content:
                    parts = content.split("Assistant:")
                    if len(parts) > 1:
                        content = parts[-1].strip()
            
            return type('Response', (), {
                'content': content
            })()
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise
    
    def _messages_to_groq_format(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Convert LangChain messages to Groq format."""
        groq_messages = []
        
        for message in messages:
            # Check if it's a LangChain message object
            if hasattr(message, 'content'):
                content = message.content
                # Determine message role
                role = "user"  # default
                
                if hasattr(message, '__class__'):
                    class_name = message.__class__.__name__
                    if 'SystemMessage' in class_name or 'system' in class_name.lower():
                        role = "system"
                    elif 'HumanMessage' in class_name or 'human' in class_name.lower():
                        role = "user"
                    elif 'AIMessage' in class_name or 'ai' in class_name.lower():
                        role = "assistant"
                
                # Also check type attribute if available
                if hasattr(message, 'type'):
                    msg_type = message.type
                    if msg_type == 'system':
                        role = "system"
                    elif msg_type == 'human':
                        role = "user"
                    elif msg_type == 'ai':
                        role = "assistant"
                
                groq_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # If it's already a dict or string, treat as user message
                if isinstance(message, dict):
                    groq_messages.append(message)
                else:
                    groq_messages.append({
                        "role": "user",
                        "content": str(message)
                    })
        
        return groq_messages


class SentenceTransformerEmbeddings:
    """Sentence Transformer embeddings wrapper."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize sentence transformer embeddings (lazy loading)."""
        self.model_name = model_name
        self.model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer model: {self.model_name} (this may take a moment...)")
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("Sentence transformer model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
            self._initialized = True
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            self.model = None
            self._initialized = True
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents."""
        self._ensure_initialized()
        if self.model:
            # Run in executor since it's sync
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, show_progress_bar=False).tolist()
            )
            return embeddings
        else:
            # Return dummy embeddings if model not available
            import numpy as np
            return [np.random.rand(384).tolist() for _ in texts]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sync embed documents."""
        self._ensure_initialized()
        if self.model:
            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
            return embeddings
        else:
            # Return dummy embeddings if model not available
            import numpy as np
            return [np.random.rand(384).tolist() for _ in texts]
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query."""
        return (await self.aembed_documents([text]))[0]
    
    def embed_query(self, text: str) -> List[float]:
        """Sync embed query."""
        return self.embed_documents([text])[0]


def get_llm(model_name: Optional[str] = None, temperature: float = 0.7):
    """Get LLM instance based on provider."""
    if settings.LLM_PROVIDER == "groq":
        return GroqLLM(
            model_name=model_name or settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature
        )
    elif settings.LLM_PROVIDER == "huggingface":
        return HuggingFaceLLM(
            model_name=model_name or settings.LLM_MODEL,
            api_key=settings.HF_TOKEN,
            temperature=temperature
        )
    elif settings.LLM_PROVIDER == "gemini":
        return GeminiLLM(
            model_name=model_name or settings.LLM_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=temperature
        )
    else:
        # Default to Groq
        logger.warning(f"Unknown LLM provider: {settings.LLM_PROVIDER}, defaulting to Groq")
        return GroqLLM(
            model_name=model_name or settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature
        )


def get_embeddings(model_name: Optional[str] = None) -> Optional[SentenceTransformerEmbeddings]:
    """Get embeddings instance."""
    return SentenceTransformerEmbeddings(
        model_name=model_name or settings.EMBEDDING_MODEL
    )

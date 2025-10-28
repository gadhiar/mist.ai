import re
from typing import List, Dict, Any, Optional
import torch
from vllm import LLM, SamplingParams

class LLMInterface:
    def __init__(self, model_path: str, max_tokens: int = 8192, n_threads: int = 8, gpu_layers: int = -1):
        """Initialize the LLM interface using VLLM with a given model.
        
        Args:
            model_path (str): Path to the model or HuggingFace model name
            max_tokens (int, optional): Maximum context length. Defaults to 8192.
            n_threads (int, optional): Number of CPU threads. Defaults to 8.
            gpu_layers (int, optional): Not used in VLLM, maintained for API compatibility.
        """
        # VLLM configuration
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Adjust based on number of GPUs available
            gpu_memory_utilization=0.6,
            max_model_len=max_tokens,
            swap_space=0,
            trust_remote_code=True,
            dtype=torch.float16,
        )
        
        # Store configuration for reference
        self.config = {
            "model_path": model_path,
            "max_tokens": max_tokens,
        }
        
    def trim_to_last_sentence(self, text: str) -> str:        
        """
        Return *text* truncated at the final full sentence boundary.
        A boundary is considered to be any '.', '!' or '?' followed by
        optional quotes/brackets, optional whitespace, and then end-of-string.

        If no sentence terminator exists, the original text is returned.
        """
        # Regex explanation:
        #   (.*?[.!?]["')\]]?)   any text lazily until a terminator
        #   \s*$                 followed only by whitespace till end-of-string
        m = re.match(r"^(.*?[.!?][\"')\]]?)\s*$", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fall back to manual search (handles cases with additional text)
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                return text[: i + 1].strip()
        return text.strip()
    
    def generate_response(self, system_prompt: str, user_message: str, conversation_history: str = "") -> str:
        """Generate a response from the LLM using chat-style prompt formatting.
        
        Args:
            system_prompt (str): The system prompt/instructions
            user_message (str): The user's input message
            conversation_history (str, optional): Any prior conversation context. Defaults to "".
            
        Returns:
            str: The generated response
        """
        # Format prompt following chat template structure
        prompt = f"""<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>
        {conversation_history}
        <|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n"""
        
        # Define sampling parameters (equivalent to the previous implementation)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=100,
            repetition_penalty=1.2,
            top_k=200,
            stop=["</s>", "<|endoftext|>", "<<USR>>", "<</USR>>", "<</SYS>>", 
                  "<</USER>>", "<</ASSISTANT>>", "<|end_header_id|>", "<<ASSISTANT>>", 
                  "<|eot_id|>", "<|im_end|>", "user:", "User:", "user :", "User :"]
        )
        
        # Generate response using VLLM
        outputs = self.llm.generate(prompt, sampling_params)
        
        # Extract and return the generated text
        if outputs and len(outputs) > 0:
            text = outputs[0].outputs[0].text
            return self.trim_to_last_sentence(text)
        return ""
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using VLLM's tokenizer.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[int]: List of token IDs
        """
        # VLLM doesn't expose tokenizer directly in the same way
        # We can access the tokenizer through the LLM instance
        tokenizer = self.llm.get_tokenizer()
        return tokenizer.encode(text)
    
    def get_token_count(self, text: str) -> int:
        """Return token count of the input text.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        return len(self.tokenize(text))
    
    def batch_generate(self, prompts: List[Dict[str, str]], 
                       max_tokens: int = 512, 
                       temperature: float = 0.7) -> List[str]:
        """Generate responses for multiple prompts in a batch.        
        Args:
            prompts (List[Dict[str, str]]): List of prompt dictionaries, each with 
                                           'system', 'user' and optional 'history' keys
            max_tokens (int, optional): Maximum tokens to generate per response
            temperature (float, optional): Temperature for sampling
            
        Returns:
            List[str]: Generated responses
        """
        formatted_prompts = []
        
        # Format each prompt according to the chat template
        for p in prompts:
            system = p.get("system", "")
            user = p.get("user", "")
            history = p.get("history", "")
            
            formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>
            {history}
            <|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>\n"""
            
            formatted_prompts.append(formatted_prompt)
        
        # Set up batch sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            repetition_penalty=1.2,
            top_k=400,
            stop=["</s>", "<|endoftext|>", "<<USR>>", "<</USR>>", "<</SYS>>", 
                  "<</USER>>", "<</ASSISTANT>>", "<|end_header_id|>", "<<ASSISTANT>>", 
                  "<|eot_id|>", "<|im_end|>", "user:", "User:", "user :", "User :"]
        )
        
        # Generate responses for all prompts in a batch
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # Extract and return the generated texts
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")
                
        return results
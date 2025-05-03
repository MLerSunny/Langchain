from langchain_ollama import ChatOllama
import time

def main():
    # Use deepseek-r1:32b model via Ollama
    model_name = "deepseek-r1:32b"
    
    print(f"Initializing Ollama with model: {model_name}")
    llm = ChatOllama(model=model_name, temperature=0.7)
    
    # Get user input for testing
    while True:
        user_input = input("\nEnter a prompt (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        print(f"\nGenerating response...")
        start_time = time.time()
        
        # Generate response
        response = llm.invoke(user_input)
        
        # Print the result
        elapsed_time = time.time() - start_time
        print(f"\nGenerated response ({elapsed_time:.2f}s):")
        print(response.content)

if __name__ == "__main__":
    main() 
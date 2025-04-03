# curl http://192.168.1.132:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "llama-3.2-1b-instruct",
# "messages": [{"role": "user", "content": "Write a python implementation of the battleships game."}],
# "temperature": 0.7
# }'


import openai
import textwrap

DEBUG = 0

global_model = "llama-3.2-1b-instruct"
global_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
global_temp = 0.7

def dprint(s):
    if DEBUG == 1:
        print(s)


def send_to_AI(text_to_AI):
    LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed
    
    global_messages.append( {"role": "user", "content": text_to_AI} )

    # Set up the OpenAI-compatible client
    client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')

    # Define the request parameters
    dprint(f"Sending AI: `{text_to_AI[:20]} ... {text_to_AI[-20:]}`")
    response = client.chat.completions.create(
        model=global_model,  # Adjust the model name based on what's running in LM Studio
        messages=global_messages,
        temperature=global_temp
    )
    resp_text = response.choices[0].message.content

    global_messages.append( {"role": "assistant", "content": resp_text} )

    dprint(f"Recving AI: `{resp_text[:20]} ... {resp_text[-20:]}`")
    dprint(global_messages)

    return resp_text




def main():
    print("\nWelcome to the Chatbot Interface! Type your message below.\n")
    print("(Press Enter on a blank line to send your message.)\n")

    
    while True:
        user_input_lines = []
        print("You: ", end="")
        while True:
            line = input()
            if line.strip() == "":  # Stop input on blank line
                break
            user_input_lines.append(line)
        
        user_text = "\n".join(user_input_lines)
        
        if user_text.startswith("//"):  # Escape sequence to send a '/'
            user_text = user_text[1:]
        elif user_text.startswith("/"):
            command_parts = user_text[1:].split(" ", 1)
            command = command_parts[0].upper()
            argument = command_parts[1] if len(command_parts) > 1 else None
            
            if command == "RESTART":
                global_messages.clear()
                global_messages.append( {"role": "system", "content": "You are a helpful AI assistant."} )
                print("Chat has been restarted.\n")
                dprint(global_messages)
                continue
            elif command == "HELP":
                print(f"Chatbot supports the following escape commands:")
                print(f"  /HELP - Displays this help") 
                print(f"  /RESTART - Restarts the chat from scratch") 
                print(f"  /DELETE - Deletes the previous chat item")
                print(f"  /MODEL <model_name> - Switches the LLM model to <model_name>.  If no model is specified, it will list available models")
                print(f"  /TEMP <temperature> - Sets the temperature value for the LLM.  Use floating point value")
                print(f"  /QUIT - Quits the chat bot")
                continue
            elif command == "MODEL":
                if argument:
                    global_model = argument
                    print(f"Chatbot model changed to: {argument}\n")
                else:
                    # List models
                    pass
                continue
            elif command == "QUIT":
                print("Goodbye! Thanks for chatting.")
                break
            elif command == "DELETE":
                if len(global_messages) > 1:
                    global_messages.pop()
                    print("Last message deleted.\n")
                continue
            elif command == "TEMP" and argument:
                try:
                    global_temp = float(argument)  # INSERT CODE HERE: Set chatbot temperature
                    print(f"Chatbot temperature set to: {global_temp}\n")
                except ValueError:
                    print("Invalid temperature value. Please enter a valid number.\n")
                continue
            else:
                print("Unknown command.\n")
                continue

    
        bot_response = send_to_AI(user_text)  # Call external chatbot function
        
        print("\nChatbot:")
        wrapped_response = textwrap.fill(bot_response, width=70)
        wrapped_response = bot_response
        print(wrapped_response)
        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()


import gradio as gr # Importing the Gradio library for creating interfaces
import openai, config, subprocess # Importing OpenAI, config, and subprocess libraries
# openai.api_key = config.OPENAI_API_KEY Setting the OpenAI API key
from config import OpenAIConfig
openai.api_key = OpenAIConfig.API


# Create a list of messages, initially containing a system message with instructions for the user
messages = [{"role": "system", "content": 'genius application developer able to easily and shortly explain web problems. Limit response by 20 words.'}]

# Define a function to transcribe user speech to text and send it to OpenAI's GPT-3.5 model
def transcribe(audio):
    global messages # Access the global variable "messages" defined earlier

    # Open the audio file from the user and transcribe it to text using OpenAI's Audio.transcribe() method
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Add the user's transcribed message to the list of messages
    messages.append({"role": "user", "content": transcript["text"]})

    # Send the list of messages to OpenAI's ChatCompletion.create() method and get a response from the GPT-3.5 model
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    # Get the system's response message from the response dictionary and add it to the list of messages
    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    # Speak the system's response aloud using the subprocess library's call() method
    subprocess.call(["say", system_message['content']])

    # Create a string variable to store the chat transcript (all messages in the messages list except system messages)
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    # Return the chat transcript
    return chat_transcript

# Create a Gradio interface with a microphone input and text output using the transcribe function defined above
ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text").launch()

# Launch the Gradio interface
ui.launch()

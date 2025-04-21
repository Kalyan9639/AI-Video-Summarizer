from google.genai import Client,types

with open("C:/Users/madda/Desktop/coding/Hackattack - 25/transcription.txt", "r") as f:
    transcript = f.read()

client = Client(api_key="AIzaSyCQIYPtpOxSorTaLhsVvQuG1_ZHD-nZct4")

chat = client.chats.create(
    model="gemini-2.0-flash",
    history = [],
    config = types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=1048,
        top_p=0.8,
        system_instruction=f"""
            You are a good communicator who can answer questions from the given text and provide information in clear and concise manner.
            The text is: {transcript}
            Use the text to answer the user's questions and provide information.
        """
    )
    )

response = chat.send_message("summarize the video transcipt that has been provided ")
print(response.text)
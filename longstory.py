# curl http://192.168.1.132:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "llama-3.2-1b-instruct",
# "messages": [{"role": "user", "content": "Write a python implementation of the battleships game."}],
# "temperature": 0.7
# }'


import openai
import tiktoken

story_start = """Once upon a time, in a valley hidden between mountains that touched the sky, there was a kingdom called Eldoria. The kingdom flourished with golden wheat fields, shimmering rivers, and forests so deep and green that they seemed to hum with ancient magic. In the heart of this kingdom stood a grand castle, its spires piercing the clouds, home to the kind and just King Edric and his daughter, Princess Lysara.

Lysara was not like other princesses. While she possessed grace and beauty, it was her insatiable curiosity that set her apart. She would often slip away from the castle, venturing into the market to speak with merchants, listen to the tales of travelers, and learn the secrets of the land from the wise old herbalist who lived at the edge of the city. But her greatest love was the stories of the Enchanted Forest, a place whispered about in hushed tones—where the trees had voices, the rivers carried messages, and the air shimmered with unseen enchantments.

No one dared enter the Enchanted Forest, for it was said to be the home of spirits and ancient beings who did not take kindly to intruders. Yet, the stories spoke of wonders beyond imagination: flowers that glowed like stars, birds that sang in forgotten tongues, and hidden treasures waiting to be found. Lysara, with her boundless curiosity, longed to see these wonders for herself.

One fateful evening, as the sun dipped below the mountains and bathed the kingdom in hues of gold and crimson, Lysara stood at her balcony, gazing at the distant treetops of the Enchanted Forest. A whisper, carried by the evening breeze, reached her ears.

“Come, seek what is lost… come, seek what is lost…”

She turned sharply, expecting to find a messenger, but there was no one there. The voice had come from the wind itself. Lysara’s heart quickened. Could it be the spirits of the forest calling her? Was this a sign that she was meant to go?

The thought consumed her. That night, long after the castle had gone to sleep, Lysara donned a cloak, tucked a small dagger into her belt, and slipped out of the castle grounds. The moon was full, casting silver light upon the land as she made her way toward the forbidden forest. The stories had always said that those who entered without permission would never return, but something deep inside urged her forward.

As she crossed the threshold into the trees, the air grew thick with the scent of wildflowers and damp earth. The forest was eerily quiet, save for the occasional rustle of leaves. The path before her twisted and turned, vanishing into the shadows. Still, she pressed on, her heart pounding with excitement.

After what felt like hours, she reached a clearing where the grass sparkled like diamonds under the moonlight. In the center of the clearing stood an ancient oak, its branches twisted with age and power. Hanging from one of the lowest branches was a lantern, but it was no ordinary lantern—it glowed with an inner light, flickering like a captured star.

As Lysara stepped closer, the lantern trembled and then, to her astonishment, spoke. “You have come, as I knew you would.”

She gasped, taking a step back. “Who… what are you?”

“I am the Lantern of Everlight,” it replied, its voice warm yet tinged with sorrow. “I was placed here long ago, to guide the one who would mend what has been broken.”

“Mend what?” Lysara asked, curiosity overcoming her fear.

“The balance of this land,” the lantern whispered. “Once, the Enchanted Forest and the kingdom of Eldoria lived in harmony, but a terrible curse divided them. The heart of the forest was stolen, its guardian imprisoned. Without the heart, the magic of this land fades, and soon, both the forest and the kingdom will wither.”

Lysara felt a chill run through her. She had heard nothing of this in the stories. “Where is the heart? How can I help?”

“The heart lies beyond the Veil of Mists, in the ruins of the Forgotten Keep. But be warned—those who took it still guard it, and they will not let it go easily.”

Lysara squared her shoulders. She had always longed for adventure, and now fate had placed one before her. “Then I shall go,” she declared. “Tell me what I must do.”

The lantern glowed brighter. “Seek the Silver Stag in the northern glade. He will show you the path.”

With that, the lantern dimmed, and the clearing returned to silence. Lysara took a deep breath. There was no turning back now.

Gathering her courage, she stepped forward, deeper into the enchanted night."""

DEBUG = 1

def dprint(s):
    if DEBUG == 1:
        print(s)

def get_num_tokens(text):
    encoding = tiktoken.get_encoding("r50k_base")

    # Encode the text into tokens
    tokens = encoding.encode(text)
    n = len(tokens)
    dprint(f"get_num_tokens returns {n}")

    return n

def send_to_AI(text_to_AI):
    LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed

    # Set up the OpenAI-compatible client
    client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')

    # Define the request parameters
    dprint(f"Sending AI: `{text_to_AI[:20]} ... {text_to_AI[-20:]}`")
    response = client.chat.completions.create(
        model="llama-3.2-1b-instruct",  # Adjust the model name based on what's running in LM Studio
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text_to_AI}
        ],
        temperature=0.7
    )
    resp_text = response.choices[0].message.content

    dprint(f"Recving AI: `{resp_text[:20]} ... {resp_text[-20:]}`")

    return resp_text
   

# Default context window size is 8192
CONTEXT_WINDOW_SIZE = 8192

# This program uses AI to grow a VERY long story with a context window of only 8K.  
# It does this by generating the story in chunks and as those chunks grow too large,
# It will use a separate AI to summarize the text into no more than 4K of tokens.  
# From there it can begin growing again.
def main():
    full_story_list = [story_start]
    context_list = [story_start]

    # [SEGMENT1] [SEGMENT2] [SEGMENT3] [SEGMENT4] ... [SEGMENT N]
    #   summarize N-3
    # [SUMMARY OF 1 thru N-3] [SEGMENT N-2] [SEGMENT N-1] [SEGMENT N]

    for i in range(100):
        dprint(f" ==== STARTING ITERATION {i} ====")
        context = "\n".join(context_list)
        if get_num_tokens(context) > (CONTEXT_WINDOW_SIZE/2):
            dprint(f"\tTRIGGERING SUMMARIZE")
            # We want to summarize 
            to_be_summarized = "\n".join(context_list[:-3])
            text_to_AI = "Please summarize the following story into no more than 1000 words.  Be sure to capture important details and events.  Really consider what kinds of information may be needed to continue writing the story from this point when writing your summary.  Do NOT respond with ANYTHING except the summary itself.\n\n" + to_be_summarized
            summary = send_to_AI(text_to_AI)
            context_list = [summary] + context_list[3:]

        text_to_AI = "Please continue the following story by adding another segment to it of approximately 1000 words.  Do NOT respond with anything except the next segment of the story.  Do NOT ask any questions.  Use all information already included in the story as well as your own imagination:\n\n" + "\n".join(context_list)
        next_segment = send_to_AI(text_to_AI)
        dprint(f"NEXT_SEGMENT: {next_segment}\n")

        full_story_list.append(next_segment)
        context_list.append(next_segment)

    print("------======== STORY ========-----")
    for segment in full_story_list:
        print(segment)
        print('  ~~~')


if __name__ == "__main__":
    main()

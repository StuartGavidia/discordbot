import nest_asyncio
import random
nest_asyncio.apply()
import discord

from utils.loaders import load_mnist, load_model
from models.AE import Autoencoder
import matplotlib.pyplot as plt
import numpy as np
import os

intents = discord.Intents.default()
intents.message_content = True
TOKEN = " " #  token for bot

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    for guild in client.guilds:
        print(
            f'{client.user} is connected to the following guild:\n'
            f'{guild.name}(id: {guild.id})'
        )

@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hi {member.name}, welcome to my Discord server!'
    )

    #if message.author == client.user:
        #return
    

@client.event
async def on_message(message):
    bot_quotes = [
        'Hello, I am a bot, � emoji.',
        'Bingo!',
        ('happy, cool, happy, cool, happy, cool '
         'no doubt no doubt no doubt no doubt.')
    ]
    #print("get a new message")
    #response = random.choice(bot_quotes)

    #print(message.content)
    #if message.content.startswith("wow"):
    #    await message.channel.send(response)
    if str(message.content) == 'wow!':
        response = random.choice(bot_quotes)
        await message.channel.send(response)
    elif message.content.startswith('calculate'):       
        expression = message.content.replace('calculate', '').strip()
        person_name = message.author.name

        try:
            print(expression)
            result = eval(expression)
            await message.channel.send(f"Hey {person_name}: The result is: {result}")
        except Exception as e:
            await message.channel.send(f"Error in calculation: {e}")
    elif message.content.startswith("generate number"):
        message_content = message.content.lower().strip()  
        parts = message_content.split()
        if len(parts) >= 3 and parts[1].lower() == "number" and parts[2].isdigit():
            digit = int(parts[2]) 
        else:
            digit = 1  

        #ensure the digit is within the valid range for MNIST (0-9)
        if digit not in range(10):
            await message.channel.send("Please provide a valid digit (0-9).")
            return

        SECTION = 'vae'
        RUN_ID = '0001'
        DATA_NAME = 'digits'
        RUN_FOLDER = 'run/{}/'.format(SECTION)
        RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

        (x_train, y_train), (x_test, y_test) = load_mnist()

        AE = load_model(Autoencoder, RUN_FOLDER)

        desired_digits = [digit]

        indices = []
        for d in desired_digits:  
            idx = np.where(y_test == d)[0]
            indices.append(idx[0])  

        example_images = x_test[indices]

        z_points = AE.encoder.predict(example_images)
        reconst_images = AE.decoder.predict(z_points)

        fig = plt.figure(figsize=(15, 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        fig = plt.figure(figsize=(5, 5)) 

        for i in range(len(indices)):
            img = reconst_images[i].squeeze()
            ax = fig.add_subplot(1, len(indices), i+1)
            ax.axis('off')
            ax.imshow(img, cmap='gray_r')

        plt.tight_layout() 

        image_path = os.path.join(RUN_FOLDER, f"images/generated_number_{digit}.png")
        fig.savefig(image_path)
        plt.close(fig)

        with open(image_path, 'rb') as f:
            picture = discord.File(f)
            await message.channel.send(file=picture)
        
client.run(TOKEN)
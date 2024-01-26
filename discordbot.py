import nest_asyncio
import random
nest_asyncio.apply()
import discord

from discord import File
from utils.loaders import load_mnist, load_model
from keras.models import load_model as load_model_keras
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from models.AE import Autoencoder
from models.GAN import GAN
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json

import openai
import io
import requests
from keras.preprocessing.text import Tokenizer

intents = discord.Intents.default()
intents.message_content = True
TOKEN = " " #token for bot

openai.api_key = ''

client = discord.Client(intents=intents)

authorized_users = ['']

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
        f'Hi {member.name}, welcome to the Discord server!'
    )

@client.event
async def on_message(message):
    if message.content.startswith('!calculate'):
        await calculate_expression(message)
    elif message.content.startswith("!generateVAE number"):
        await generate_vae_number(message)
    elif message.content.startswith("!generateGAN horse"):
        await generate_gan_horse(message)
    elif message.content.startswith("!generateGAN person"):
        await generate_gan_person(message)
    elif message.content.startswith("!chatLSTM"):
        prompt = message.content[len('!chatLSTM '):] #extract prompt
        await generate_chat_response(message)
    elif message.content.startswith('!chatgpt'):
        prompt = message.content[len('!chatgpt '):] #extract prompt
        response = await chatgpt_response(message)
        if len(response) > 2000:
            parts = split_long_message(response)
            for part in parts:
                await message.channel.send(part)
        else:
            await message.channel.send(response)
    elif message.content.startswith('!dalle'):
        prompt = message.content[len('!dalle '):] #extract prompt
        image_bytes = await generate_dalle_image(prompt)
        if image_bytes:
            await message.channel.send(file=File(image_bytes, 'dalle_image.png'))
        else:
            await message.channel.send("Sorry, I couldn't generate an image for that prompt.")
    elif message.content == "!terminate":
        if str(message.author.id) in authorized_users:
            await message.channel.send("Shutting down...")
            await client.close()
        else:
            await message.channel.send("You do not have permission to use this command.")
    elif message.content == "!help":
        await help_command(message)

async def help_command(message):
    help_text = (
        "Available Commands:\n"
        "!calculate <expression> - Calculates the result of a mathematical expression.\n"
        "!generateVAE number <digit> - Generates a digit image using a VAE model.\n"
        "!generateGAN horse - Generates an image of a horse using a GAN model.\n"
        "!generateGAN person - Generates an image of a person using a GAN model.\n"
        "!chatgpt <prompt> - Interacts with the ChatGPT model to generate a text response.\n"
        "!dalle <prompt> - Generates an image using DALL-E 3 based on the given prompt.\n"
        "!terminate - Shuts down the bot (restricted to authorized users).\n"
    )

    await message.channel.send(help_text)



async def calculate_expression(message):
    expression = message.content.replace('!calculate', '').strip()
    person_name = message.author.name

    try:
        print(expression)
        result = eval(expression)
        await message.channel.send(f"Hey {person_name}: The result is: {result}")
    except Exception as e:
        await message.channel.send(f"Error in calculation: {e}")

async def generate_vae_number(message):
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

async def generate_gan_horse(message):
    def generate_images(generator, num_images):
        random_latent_vectors = np.random.normal(size=(num_images, gan.z_dim))

        generated_images = generator.predict(random_latent_vectors)

        return generated_images

    SECTION = 'gan'
    RUN_ID = '0001'
    DATA_NAME = 'horse'
    RUN_FOLDER = 'run/{}/'.format(SECTION)
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    gan = GAN(input_dim = (28, 28, 3)  
    , discriminator_conv_filters = [64, 64, 128, 128]
    , discriminator_conv_kernel_size = [5, 5, 5, 5]
    , discriminator_conv_strides = [2, 2, 2, 1]
    , discriminator_batch_norm_momentum = None
    , discriminator_activation = 'relu'
    , discriminator_dropout_rate = 0.4
    , discriminator_learning_rate = 0.0008
    , generator_initial_dense_layer_size = (7, 7, 64)
    , generator_upsample = [2, 2, 1, 1]
    , generator_conv_filters = [128, 64, 64, 3]  
    , generator_conv_kernel_size = [5, 5, 5, 5]
    , generator_conv_strides = [1, 1, 1, 1]
    , generator_batch_norm_momentum = 0.9
    , generator_activation = 'relu'
    , generator_dropout_rate = None
    , generator_learning_rate = 0.0004
    , optimiser = 'rmsprop'
    , z_dim = 100
    )

    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    num_images = 1
    generated_images = generate_images(gan.generator, num_images)
    generated_image = (generated_images[0] + 1) * 127.5
    generated_image = np.clip(generated_image, 0, 255).astype('uint8')

    upscaled_image = Image.fromarray(generated_image).resize((256, 256))  # Example size

    image_path = os.path.join(RUN_FOLDER, 'generated_horse.png')
    upscaled_image.save(image_path)

    await message.channel.send(file=discord.File(image_path))

async def generate_gan_person(message):
    def generate_images(generator, num_images):
        random_latent_vectors = np.random.normal(size=(num_images, gan.z_dim))

        generated_images = generator.predict(random_latent_vectors)

        return generated_images

    SECTION = 'gan'
    RUN_ID = '0001'
    DATA_NAME = 'face'
    RUN_FOLDER = 'run/{}/'.format(SECTION)
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    gan = GAN(input_dim = (28, 28, 3)  
    , discriminator_conv_filters = [64, 64, 128, 128]
    , discriminator_conv_kernel_size = [5, 5, 5, 5]
    , discriminator_conv_strides = [2, 2, 2, 1]
    , discriminator_batch_norm_momentum = None
    , discriminator_activation = 'relu'
    , discriminator_dropout_rate = 0.4
    , discriminator_learning_rate = 0.0008
    , generator_initial_dense_layer_size = (7, 7, 64)
    , generator_upsample = [2, 2, 1, 1]
    , generator_conv_filters = [128, 64, 64, 3]  
    , generator_conv_kernel_size = [5, 5, 5, 5]
    , generator_conv_strides = [1, 1, 1, 1]
    , generator_batch_norm_momentum = 0.9
    , generator_activation = 'relu'
    , generator_dropout_rate = None
    , generator_learning_rate = 0.0004
    , optimiser = 'rmsprop'
    , z_dim = 100
    )

    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    num_images = 1
    generated_images = generate_images(gan.generator, num_images)
    generated_image = (generated_images[0] + 1) * 127.5
    generated_image = np.clip(generated_image, 0, 255).astype('uint8')

    upscaled_image = Image.fromarray(generated_image).resize((256, 256))  # Example size

    image_path = os.path.join(RUN_FOLDER, 'generated_face.png')
    upscaled_image.save(image_path)

    await message.channel.send(file=discord.File(image_path))

async def generate_chat_response(message):
    model = load_model_keras('./saved_models/aesop_dropout_100.h5')
    next_words = 1
    temp = 0.2
    max_sequence_len = 1  # Consider increasing this if it's feasible for your model
    start_story = '| ' * max_sequence_len

    seed_text = message.content
    output_text = seed_text
    seed_text = start_story + seed_text

    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if not token_list:
            break

        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')[0]

        probs = model.predict(np.array([token_list]), verbose=0)[0]
        y_class = sample_with_temp(probs, temperature=temp)

        if y_class == 0:
            break

        output_word = tokenizer.index_word.get(y_class, '')

        if output_word == "|" or not output_word:
            break

        output_text += output_word + ' '
        seed_text = output_text[-max_sequence_len:]

    await message.channel.send(output_text)
    

def sample_with_temp(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

async def chatgpt_response(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[{"role": "system", "content": "You are a discord bot."},
                      {"role": "user", "content": message.content}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I couldn't process that request."

def split_long_message(message, max_length=2000):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

async def generate_dalle_image(prompt):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response['data'][0]['url']  

        image_bytes = io.BytesIO(requests.get(image_url).content)
        return image_bytes
    except Exception as e:
        print(f"Error: {e}")
        return None

client.run(TOKEN)
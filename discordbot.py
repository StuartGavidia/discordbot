import nest_asyncio
import random
nest_asyncio.apply()
import discord

from utils.loaders import load_mnist, load_model
from models.AE import Autoencoder
from models.GAN import GAN
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

intents = discord.Intents.default()
intents.message_content = True
TOKEN = " MTE3OTE1MjU2NDg0ODE3MzA2Ng.GkogGL.4AwfXI4PZYx9za_dFxOn7b7UJMhaAqYwLzg6xQ" #token for bot

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
    elif message.content == "!help":
        await help_command(message)

async def help_command(message):
    help_text = (
        "Available Commands:\n"
        "!calculate <expression> - Calculates the result of a mathematical expression.\n"
        "!generateVAE number <digit> - Generates a digit image using a VAE model.\n"
        "!generateGAN horse - Generates an image of a horse using a GAN model.\n"
    )
    await message.channel.send(help_text)

async def calculate_expression(message):
    expression = message.content.replace('calculate', '').strip()
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

client.run(TOKEN)
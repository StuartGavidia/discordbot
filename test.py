import nest_asyncio
import random
nest_asyncio.apply()
import discord
intents = discord.Intents.default()
intents.message_content = True
TOKEN = " MTE3OTE1MjU2NDg0ODE3MzA2Ng.GCEuC4.ro3pKi-xlqPviduSux6lk5ZeaJ-cIF15rqcSWo" # you
#should copy your token for your bot
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
        
client.run(TOKEN)
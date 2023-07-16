import pygame
import random
import button
import time
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import gym
from gym import spaces
from collections import namedtuple, deque
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
TOTAL_TS = 100
SYNC_FREQ = 10
seed = 32

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)


# Set the seed value
np.random.seed(seed)
random.seed(seed)

pygame.init()

clock = pygame.time.Clock()
fps = 200

#game window
bottom_panel = 150
screen_width = 800
screen_height = 400 + bottom_panel

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Battle')


#define game variables
current_fighter = 0
total_fighters = 3
action_cooldown = 0
action_wait_time = 20 # 小さくすると早くなる
attack = False
potion = False
potion_effect = 15
clicked = False
game_over = 0


#define fonts
font = pygame.font.SysFont('Times New Roman', 26)

#define colours
red = (255, 0, 0)
green = (0, 255, 0)

#load images
#background image
background_img = pygame.image.load('img/Background/background.png').convert_alpha()
#panel image
panel_img = pygame.image.load('img/Icons/panel.png').convert_alpha()
#button images
potion_img = pygame.image.load('img/Icons/potion.png').convert_alpha()
restart_img = pygame.image.load('img/Icons/restart.png').convert_alpha()
#load victory and defeat images
victory_img = pygame.image.load('img/Icons/victory.png').convert_alpha()
defeat_img = pygame.image.load('img/Icons/defeat.png').convert_alpha()
#sword image
sword_img = pygame.image.load('img/Icons/sword.png').convert_alpha()


#create function for drawing text
def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))


#function for drawing background
def draw_bg():
	screen.blit(background_img, (0, 0))


#function for drawing panel
def draw_panel():
	#draw panel rectangle
	screen.blit(panel_img, (0, screen_height - bottom_panel))
	#show knight stats
	draw_text(f'{knight.name} HP: {knight.hp}', font, red, 100, screen_height - bottom_panel + 10)
	for count, i in enumerate(bandit_list):
		#show name and health
		draw_text(f'{i.name} HP: {i.hp}', font, red, 550, (screen_height - bottom_panel + 10) + count * 60)




#fighter class
class Fighter():
	def __init__(self, x, y, name, max_hp, strength, potions):
		self.name = name
		self.max_hp = max_hp
		self.hp = max_hp
		self.strength = strength
		self.start_potions = potions
		self.potions = potions
		self.alive = True
		self.animation_list = []
		self.frame_index = 0
		self.action = 0#0:idle, 1:attack, 2:hurt, 3:dead
		self.update_time = pygame.time.get_ticks()
		#load idle images
		temp_list = []
		for i in range(8):
			img = pygame.image.load(f'img/{self.name}/Idle/{i}.png')
			img = pygame.transform.scale(img, (img.get_width() * 3, img.get_height() * 3))
			temp_list.append(img)
		self.animation_list.append(temp_list)
		#load attack images
		temp_list = []
		for i in range(8):
			img = pygame.image.load(f'img/{self.name}/Attack/{i}.png')
			img = pygame.transform.scale(img, (img.get_width() * 3, img.get_height() * 3))
			temp_list.append(img)
		self.animation_list.append(temp_list)
		#load hurt images
		temp_list = []
		for i in range(3):
			img = pygame.image.load(f'img/{self.name}/Hurt/{i}.png')
			img = pygame.transform.scale(img, (img.get_width() * 3, img.get_height() * 3))
			temp_list.append(img)
		self.animation_list.append(temp_list)
		#load death images
		temp_list = []
		for i in range(10):
			img = pygame.image.load(f'img/{self.name}/Death/{i}.png')
			img = pygame.transform.scale(img, (img.get_width() * 3, img.get_height() * 3))
			temp_list.append(img)
		self.animation_list.append(temp_list)
		self.image = self.animation_list[self.action][self.frame_index]
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)


	def update(self):
		animation_cooldown = 100
		#handle animation
		#update image
		self.image = self.animation_list[self.action][self.frame_index]
		#check if enough time has passed since the last update
		if pygame.time.get_ticks() - self.update_time > animation_cooldown:
			self.update_time = pygame.time.get_ticks()
			self.frame_index += 1
		#if the animation has run out then reset back to the start
		if self.frame_index >= len(self.animation_list[self.action]):
			if self.action == 3:
				self.frame_index = len(self.animation_list[self.action]) - 1
			else:
				self.idle()


	
	def idle(self):
		#set variables to idle animation
		self.action = 0
		self.frame_index = 0
		self.update_time = pygame.time.get_ticks()


	def attack(self, target):
		#deal damage to enemy
		rand = random.randint(-2, 2)
		damage = self.strength + rand
		# damage = self.strength
		target.hp -= damage
		#run enemy hurt animation
		target.hurt()
		#check if target has died
		if target.hp < 1:
			target.hp = 0
			target.alive = False
			target.death()
		damage_text = DamageText(target.rect.centerx, target.rect.y, str(damage), red)
		damage_text_group.add(damage_text)
		#set variables to attack animation
		self.action = 1
		self.frame_index = 0
		self.update_time = pygame.time.get_ticks()

	def hurt(self):
		#set variables to hurt animation
		self.action = 2
		self.frame_index = 0
		self.update_time = pygame.time.get_ticks()

	def death(self):
		#set variables to death animation
		self.action = 3
		self.frame_index = 0
		self.update_time = pygame.time.get_ticks()


	def reset (self):
		self.alive = True
		self.potions = self.start_potions
		self.hp = self.max_hp
		self.frame_index = 0
		self.action = 0
		self.update_time = pygame.time.get_ticks()


	def draw(self):
		screen.blit(self.image, self.rect)



class HealthBar():
	def __init__(self, x, y, hp, max_hp):
		self.x = x
		self.y = y
		self.hp = hp
		self.max_hp = max_hp


	def draw(self, hp):
		#update with new health
		self.hp = hp
		#calculate health ratio
		ratio = self.hp / self.max_hp
		pygame.draw.rect(screen, red, (self.x, self.y, 150, 20))
		pygame.draw.rect(screen, green, (self.x, self.y, 150 * ratio, 20))



class DamageText(pygame.sprite.Sprite):
	def __init__(self, x, y, damage, colour):
		pygame.sprite.Sprite.__init__(self)
		self.image = font.render(damage, True, colour)
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)
		self.counter = 0


	def update(self):
		#move damage text up
		self.rect.y -= 1
		#delete the text after a few seconds
		self.counter += 1
		if self.counter > 30:
			self.kill()



damage_text_group = pygame.sprite.Group()

# x, y, name, max_hp, strength, potions
max_knight_hp = 30 
max_num_potions = 3
max_bandit1_hp = 20
max_bandit2_hp = 5
max_current_fighter = 3

knight = Fighter(200, 260, 'Knight', max_knight_hp, 10, max_num_potions)
bandit1 = Fighter(550, 270, 'Bandit', max_bandit1_hp, 10, 1)
bandit2 = Fighter(700, 270, 'Bandit', max_bandit2_hp, 35, 1)

bandit_list = []
bandit_list.append(bandit1)
bandit_list.append(bandit2)

knight_health_bar = HealthBar(100, screen_height - bottom_panel + 40, knight.hp, knight.max_hp)
bandit1_health_bar = HealthBar(550, screen_height - bottom_panel + 40, bandit1.hp, bandit1.max_hp)
bandit2_health_bar = HealthBar(550, screen_height - bottom_panel + 100, bandit2.hp, bandit2.max_hp)

#create buttons
potion_button = button.Button(screen, 100, screen_height - bottom_panel + 70, potion_img, 64, 64)
restart_button = button.Button(screen, 330, 120, restart_img, 120, 30)



# all action list
actions_and_targets = [('attack', 0), ('attack', 1), ('potion', None)]
num_actions = len(actions_and_targets)

# ターン制バトル環境　gym.Envを継承
class BattleEnv(gym.Env):
    def __init__(self):
        super(BattleEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, Box for continuous
        self.action_space = spaces.Discrete(num_actions)  # [('attack', 0), ('attack', 1), ('potion', None)]
        self.observation_space = spaces.Tuple((
            spaces.Discrete(max_knight_hp),  # Player HP
            spaces.Discrete(max_num_potions),   # Player potion count
            spaces.Discrete(max_bandit1_hp),  # Enemy 1 HP
            spaces.Discrete(max_bandit2_hp),  # Enemy 2 HP
            spaces.Discrete(max_current_fighter)))   # Whose turn: 0 for player, 1 for enemy 1, 2 for enemy 2
	
		# Initialize state
        self.knight_hp = max_knight_hp
        self.num_potions = max_num_potions
        self.bandit1_hp = max_bandit1_hp
        self.bandit2_hp = max_bandit2_hp
        self.current_fighter = 0


    def step(self, action):
        # Execute one time step within the environment
        # This is where you define your game logic and return the next state, reward and done
        # You would also need to handle your action here
        pass

    def reset(self):
        self.knight_hp = max_knight_hp
        self.num_potions = max_num_potions
        self.bandit1_hp = max_bandit1_hp
        self.bandit2_hp = max_bandit2_hp
        self.current_fighter = 0
        return np.array([self.knight_hp, self.num_potions, self.bandit1_hp, self.bandit2_hp, self.current_fighter])

    def render(self, mode='human'):
        # Render the environment to the screen (optional)
        pass

# Transition（遷移） - （状態、アクション）のペアを（next_state、報酬）の結果にマッピング
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 過去の経験を保存するクラス
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    # input: state, output: action
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = BattleEnv()  # Use the custom environment

# Get number of actions and observations from gym action space
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

# Initialize action-value function Q with random weights
policy_net = DQN(n_observations, n_actions).to(device) # ターゲットQ値を計算するためのネットワーク
target_net = DQN(n_observations, n_actions).to(device) # 実際の行動を選択するためのネットワーク

# Initialize optimizer
optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)

# Initialize memory
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

# エージェントの行動選択
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# エピソードの結果をプロット
episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
# 最適化（学習）関数
def optimize_model():
    # 十分な経験が蓄積されていない場合、学習を行わない
    if len(memory) < BATCH_SIZE: 
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# 描画用の報酬の総和
cumulative_rewards = [0]

# Main loop for DQN
num_episodes = 50
for episode in range(num_episodes):
	run = True
	states = []
	actions = []

	while run:

		clock.tick(fps)

		# Initialize the environment and get it's state
		state= env.reset()
		state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

		#draw background
		draw_bg()

		#draw panel
		draw_panel()
		knight_health_bar.draw(knight.hp)
		bandit1_health_bar.draw(bandit1.hp)
		bandit2_health_bar.draw(bandit2.hp)

		#draw fighters
		knight.update()
		knight.draw()
		for bandit in bandit_list:
			bandit.update()
			bandit.draw()

		#draw potion
		potion_button.draw()

		#draw the damage text
		damage_text_group.update()
		damage_text_group.draw(screen)

		#control player actions
		#reset action variables
		attack = False
		potion = False
		target = None
			
		#show number of potions remaining
		draw_text(str(knight.potions), font, red, 150, screen_height - bottom_panel + 70)




		if game_over == 0:
			#player action
			if knight.alive == True:
				if current_fighter == 1:
					action_cooldown += 1
					if action_cooldown >= action_wait_time:

						# Choose an action. 確定でない（いないターゲットを選択など）ため、stateは仮置き
						state = np.array([knight.hp, knight.potions, bandit1.hp, bandit2.hp, current_fighter])
						state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
						action_id = select_action(state)
						action, target_index = actions_and_targets[action_id]

						if action == 'attack':
							# Select a bandit to attack
							target = bandit_list[target_index]
							if target.alive == True:
								attack = True
								states.append(state)
								actions.append(action_id)
								print(action, target_index)
						elif action == 'potion':
							if knight.potions > 0:
								potion = True
								states.append(state)
								actions.append(action_id)
								print(action, target_index)

						#look for player action
						#attack
						if attack == True and target != None:
							knight.attack(target)
							current_fighter += 1
							action_cooldown = 0
						#potion
						if potion == True:
							if knight.potions > 0:
								#check if the potion would heal the player beyond max health
								if knight.max_hp - knight.hp > potion_effect:
									heal_amount = potion_effect
								else:
									heal_amount = knight.max_hp - knight.hp
								knight.hp += heal_amount
								knight.potions -= 1
								damage_text = DamageText(knight.rect.centerx, knight.rect.y, str(heal_amount), green)
								damage_text_group.add(damage_text)
								current_fighter += 1
								action_cooldown = 0
						print("action_id: ", action_id)
						print("state: ", state)
			else:
				game_over = -1


			#enemy action
			for count, bandit in enumerate(bandit_list):
				if current_fighter == 1 + count:
					if bandit.alive == True:
						action_cooldown += 1
						if action_cooldown >= action_wait_time:
							#check if bandit needs to heal first
							if (bandit.hp / bandit.max_hp) < 0.5 and bandit.potions > 0:
								#check if the potion would heal the bandit beyond max health
								if bandit.max_hp - bandit.hp > potion_effect:
									heal_amount = potion_effect
								else:
									heal_amount = bandit.max_hp - bandit.hp
								bandit.hp += heal_amount
								bandit.potions -= 1
								damage_text = DamageText(bandit.rect.centerx, bandit.rect.y, str(heal_amount), green)
								damage_text_group.add(damage_text)
								current_fighter += 1
								action_cooldown = 0
							#attack
							else:
								bandit.attack(knight)
								current_fighter += 1
								action_cooldown = 0
					else:
						current_fighter += 1

			#if all fighters have had a turn then reset
			if current_fighter > total_fighters:
				current_fighter = 0


		#check if all bandits are dead
		alive_bandits = 0
		for bandit in bandit_list:
			if bandit.alive == True:
				alive_bandits += 1
		if alive_bandits == 0:
			game_over = 1


		#check if game is over
		if game_over != 0:
			if game_over == 1:
				screen.blit(victory_img, (250, 50))
				reward = 1
				print("win")
			if game_over == -1:
				screen.blit(defeat_img, (290, 50))
				reward = -1
				print("lose")
			
			# Update the display and wait for 0.2 second
			pygame.display.flip()
			pygame.time.delay(200)
			
			# At the end of the episode, add the reward to the cumulative reward
			cumulative_reward = cumulative_rewards[-1] + reward
			cumulative_rewards.append(cumulative_reward)

			knight.reset()
			for bandit in bandit_list:
				bandit.reset()
			current_fighter = 0
			action_cooldown
			game_over = 0
			action_index = 0
			break


		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				sys.exit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				clicked = True
			else:
				clicked = False

		pygame.display.update()
	
	# Backpropagate the reward
	print("reward: ", reward)
	G = 0
	for t in reversed(range(len(states))):
		G = gamma * G + reward
		Q_table[states[t], actions[t]] = (1 - alpha) * Q_table[states[t], actions[t]] + alpha * G


# After all episodes are done, plot the rewards
plt.plot(cumulative_rewards)
plt.title('Cumulative reward over time')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
# Save the figure to a file
plt.savefig('rewards.png')

#time.sleep(5)
pygame.quit()


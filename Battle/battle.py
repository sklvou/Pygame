import pygame
import random
import button
import time
import numpy as np
import sys

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
current_fighter = 1
total_fighters = 3
action_cooldown = 0
action_wait_time = 10
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
knight = Fighter(200, 260, 'Knight', 30, 10, 3)
#bandit1 = Fighter(550, 270, 'Bandit', 5, 35, 1)
#bandit2 = Fighter(700, 270, 'Bandit', 20, 12, 1)
bandit1 = Fighter(550, 270, 'Bandit', 20, 12, 1)
bandit2 = Fighter(700, 270, 'Bandit', 5, 35, 1)

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
#actions_and_targets = [('potion', None), ('attack', 0), ('attack', 1)]
actions_and_targets = [('attack', 0), ('attack', 1), ('potion', None)]
# state (HPやポーションの状態をそれぞれ2値化)
def encode_state(knight_hp, num_potions, bandit1_hp, bandit2_hp, current_fighter):
    # Binary encoding of the state variables
    knight_hp_bin = 1 if knight_hp > 15 else 0
    num_potions_bin = 1 if num_potions > 0 else 0
    bandit1_hp_bin = 1 if bandit1_hp > 10 else 0
    bandit2_hp_bin = 1 if bandit2_hp > 10 else 0

    # Combine the binary variables into one integer
    state = (knight_hp_bin * 2**4 +
             num_potions_bin * 2**3 +
             bandit1_hp_bin * 2**2 +
             bandit2_hp_bin * 2**1 +
             current_fighter)  # Assuming current_fighter is 0 or 1

    return state



# Number of states and actions
max_knight_hp = 30 
max_num_potions = 3
max_bandit_hp = 20
max_current_fighter = 3
#num_states = (max_knight_hp + 1) * (max_num_potions + 1) * (max_bandit_hp + 1) ** 2 * (max_current_fighter + 1) - 1
num_states = 32
num_actions = len(actions_and_targets)

# Initialize Q-table with zeros
Q_table = np.zeros((num_states, num_actions))

# choose action by ε-greedy
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random action
        action = np.random.choice(num_actions)
    else:
        # Choose the action with the highest Q-value for the current state
        action = np.argmax(Q_table[state])
    return action


# Parameters
alpha = 0.5  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 150  # Number of games to play

# Main loop for Q-learning
for episode in range(num_episodes):
	run = True
	states = []
	actions = []

	while run:

		clock.tick(fps)

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

						# Choose an action
						state = encode_state(knight.hp, knight.potions, bandit1.hp, bandit2.hp, current_fighter)
						action_id = choose_action(state, epsilon)

						# Get the next action and target from the predefined actions and targets
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
			else:
				game_over = -1


			#enemy action
			for count, bandit in enumerate(bandit_list):
				if current_fighter == 2 + count:
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
				current_fighter = 1


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
			#if restart_button.draw():
			knight.reset()
			for bandit in bandit_list:
				bandit.reset()
			current_fighter = 1
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

#time.sleep(5)
pygame.quit()


# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:25:00 2024

@author: Eva
"""
from pygame.locals import *
def main():
    gameChoice = input("Hi! I'm Eva! I made two games: Blowfish Jump and Blackjack. Which would you like to play (case sensitive)?")
    if gameChoice == "Blowfish Jump":
        import pygame
        import sys
        import random
        pygame.init()
        window_x = 600
        window_y = 600
        playerX = window_x/2
        playerY = window_y/2
        window = pygame.display.set_mode((window_x, window_y))
        fillcolours = (0,0,0)
        hitboxcolour = (255,0,0)
        playervX = 0
        playervY = 0
        tick = 0
        ticksperframe = 16
        framecounter = 0
        shotsSurvived = 0
        pygame.display.flip()
        Turtlestats = {"img":pygame.image.load("/home/pi/Desktop/T12024-2025/T1-2024-2025/images/Turtle.png"),"SPD":1, "HP":12, "Friction":0.1, "shotCounter": 0}
        PeacockStats = {"img": pygame.image.load("./images/Peacock.png"), "SPD": 1, "HP": 4, "Friction": 0.1, "shotCounter": 0}
        HippoStats = {"img": pygame.image.load("./images/Hippo.png"), "SPD": 0.5, "HP": 24, "Friction": 0.075, "shotCounter": 0}
        TigerStats = {"img": pygame.image.load("./images/Tiger.png"), "SPD": 1.5, "HP": 1, "Friction": 0.1, "shotCounter": 0}
        KangarooStats = {"img": pygame.image.load("./images/Kangaroo.png"), "SPD": 1, "HP": 1, "Friction": 0.1, "shotCounter": 0}
        OrcaStats = {"img": pygame.image.load("./images/Orca.png"), "SPD": 1, "HP": 1, "Friction": 0.1, "shotCounter": 0}
        HighlandCowStats = {"img": pygame.image.load("./images/Highland_Cow.png"), "SPD": 0.5, "HP": 12, "Friction": 0.075, "shotCounter": 0, "trumpets": 0}
        GoldenStats = {"img": pygame.image.load("./images/Golden_Retriever.png"), "SPD": 2, "HP": 0, "Friction": 0.1, "shotCounter": 0}
        BrainCrampStats = {"img": pygame.image.load("./images/Brain_Cramp.png"), "SPD": 70, "HP": 4, "Friction": 0, "shotCounter": 0, "Melon?": True, "jumpcounter": 0}
        charlist = [Turtlestats, PeacockStats, HippoStats, TigerStats, KangarooStats, OrcaStats, HighlandCowStats, BrainCrampStats]
        choosing = True
        print("Welcome to Blowfish Dodge! Using characters from my favourite video game, Super Auto Pets, I created a little minigame where you have to dodge the rocks thrown at you by the Blowfish. You have 7 character choices, who all come from Super Auto Pets.")
        print("The first character is the Turtle, who is your default character because it is on the app icon. It has", Turtlestats["HP"], "health and no special ability.")
        print("The second character is the Peacock, who gets faster every time it is hit. It has", PeacockStats["HP"], "health.")
        print("The third character is the Hippo, who is slower than normal but really tanky. It has", HippoStats["HP"], "health.")
        print("The fourth character is the Tiger, who is made specifically for sweats. It has a measly", TigerStats["HP"], "health, but moves faster than normal and gains points twice as fast.")
        print("The fifth character is the Kangaroo, who starts off at", KangarooStats["HP"], "health, but can heal all the way up to 12 health if you survive without getting hit long enough.")
        print("The sixth character is the Highland Cow, who is slow, but gains trumpets every 12 shots to summon a Golden Retriever with that much health after the Cow gets hit to 0 HP. The Highland Cow has", HighlandCowStats["HP"], "health.")
        print("The last (but not the least) character is the Brain Cramp, who jumps around the arena. Every once in a while, it gains the Melon perk, which prevents it from taking damage for a short time. It has", BrainCrampStats["HP"], "health.")
        print("Use the spacebar to start the game after picking your character and the arrow keys to move. Now choose your character and have fun!")
        while choosing:
            character = input("Which character would you like to play?")
            if character == "Turtle":
                character = Turtlestats
                choosing = False
            elif character == "Peacock":
                character = PeacockStats
                choosing = False
            elif character == "Hippo":
                character = HippoStats
                choosing = False
            elif character == "Tiger":
                character = TigerStats
                choosing = False
            elif character == "Kangaroo":
                character = KangarooStats
                choosing = False
            elif character == "Orca":
                character = OrcaStats
                choosing = False
            elif character == "Highland Cow":
                character = HighlandCowStats
                choosing = False
            elif character == "Brain Cramp":
                character = BrainCrampStats
                choosing = False
            else:
                print("Not a valid choice! Try again")
        healthbar = pygame.image.load("./images/New Piskel(13).png")
        healthbar = pygame.transform.scale(healthbar, (60, 60))
        character["img"] = pygame.transform.scale(character["img"], (60, 60))
        Blowstats = {"y":0,"Blowfish":pygame.image.load("./images/Blowfish.png"), "ATKSPD":5, "ProjectileStats":{"Projectile": pygame.image.load("./images/image-removebg-preview(28).png"),"DMG":1,"SPD":0.5, "x":0, "y":0, "active?":False}}
        Blowstats["Blowfish"] = pygame.transform.flip(pygame.transform.scale(Blowstats["Blowfish"], (60, 60)), True, False)
        Blowstats["ProjectileStats"]["Projectile"] = pygame.transform.scale(Blowstats["ProjectileStats"]["Projectile"], (30, 30))
        trumpet = pygame.image.load("./images/Trumpet.png")
        trumpet = pygame.transform.scale(trumpet, (30, 30))
        melon = pygame.image.load("./images/Melon.png")
        melon = pygame.transform.scale(melon, (20, 20))
        def drawHitboxes (x, y, l, h, w):
            pygame.draw.rect(window, hitboxcolour, pygame.Rect(x, y, l, h), w)
        running = True
        playing = False
        myfont = pygame.font.SysFont("monospace", 30)
        textcolour = (255, 255, 255)
        while running:
            currentHP = character["HP"]
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:            
                        playing = True
                        if character == KangarooStats:
                            character["HP"] = 12
            window.fill(fillcolours)
            window.blit(character["img"], (playerX, playerY))
            window.blit(Blowstats["ProjectileStats"]["Projectile"], (Blowstats["ProjectileStats"]["x"] - 15, Blowstats["ProjectileStats"]["y"] - 15))
            window.blit(healthbar, (playerX, playerY - 10), (0, (character["HP"] - currentHP)*5*12/character["HP"], 60, 5))
            #drawHitboxes(playerX - 30, playerY - 30, 60, 60, 3)
            #drawHitboxes(Blowstats["ProjectileStats"]["x"],Blowstats["ProjectileStats"]["y"], 5, 5, 0)
            window.blit(Blowstats["Blowfish"], (50, Blowstats["y"]))
            while playing:
                window.fill(fillcolours)
                window.blit(character["img"], (playerX, playerY))
                window.blit(Blowstats["ProjectileStats"]["Projectile"], (Blowstats["ProjectileStats"]["x"] - 15, Blowstats["ProjectileStats"]["y"] - 15))
                window.blit(healthbar, (playerX, playerY - 10), (0, (character["HP"] - currentHP)*5*12/character["HP"], 60, 5))
                #drawHitboxes(playerX - 30, playerY - 30, 60, 60, 3)
                #drawHitboxes(Blowstats["ProjectileStats"]["x"],Blowstats["ProjectileStats"]["y"], 5, 5, 0)
                window.blit(Blowstats["Blowfish"], (50, Blowstats["y"]))
                text = str(shotsSurvived)
                label = myfont.render(text, 1, textcolour)
                window.blit(label, (550, 0))
                if character == HighlandCowStats or character == GoldenStats:
                    text1 = str(HighlandCowStats["trumpets"])
                    label = myfont.render(text1, 1, textcolour)
                    window.blit(label, (60, 0))
                    window.blit(trumpet, (20, 0))
                if character == BrainCrampStats:
                    if character["shotCounter"] == 0:
                        window.blit(melon, (playerX, playerY))
                HPtext = str(currentHP)
                label = myfont.render(HPtext, 1, textcolour)
                window.blit(label, (playerX + 20, playerY - 40))
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                    if event.type == pygame.KEYDOWN: 
                        if event.key == pygame.K_LEFT:
                            playervX -= character["SPD"]
                            if character == BrainCrampStats:
                                character["jumpcounter"] += 1
                                if character["jumpcounter"] == 4:
                                    character["jumpcounter"] = 0
                                    character["shotCounter"] = 0
                        elif event.key == pygame.K_RIGHT:
                            playervX += character["SPD"]
                            if character == BrainCrampStats:
                                character["jumpcounter"] += 1
                                if character["jumpcounter"] == 4:
                                    character["jumpcounter"] = 0
                                    character["shotCounter"] = 0
                        elif event.key == pygame.K_UP:
                            playervY -= character["SPD"]
                            if character == BrainCrampStats:
                                character["jumpcounter"] += 1
                                if character["jumpcounter"] == 4:
                                    character["jumpcounter"] = 0
                                    character["shotCounter"] = 0
                        elif event.key == pygame.K_DOWN:
                            playervY += character["SPD"]
                            if character == BrainCrampStats:
                                character["jumpcounter"] += 1
                                if character["jumpcounter"] == 4:
                                    character["jumpcounter"] = 0
                                    character["shotCounter"] = 0
                tick += 1
                if framecounter == Blowstats["ATKSPD"]:
                    Blowstats["y"] = playerY
                    if Blowstats["ProjectileStats"]["active?"] == False:
                        Blowstats["ProjectileStats"]["active?"] = True
                        Blowstats["ProjectileStats"]["y"] = Blowstats["y"] + 15
                        Blowstats["ProjectileStats"]["x"] = 50
                        shotsSurvived += 1
                        character["shotCounter"] += 1
                        if character == TigerStats:
                            shotsSurvived += 1
                        if character["shotCounter"] == 12:
                            if character == KangarooStats:
                                if currentHP < 12:
                                    currentHP += 1
                                    character["shotCounter"] = 0
                            elif character == HighlandCowStats:
                                character["trumpets"] += 1
                                character["shotCounter"] = 0
                    framecounter = 0
                Blowstats["ProjectileStats"]["x"] += Blowstats["ProjectileStats"]["SPD"]
                Blowstats["ProjectileStats"]["y"] += 4*random.random() - 2
                if Blowstats["ProjectileStats"]["x"] > window_x:
                    Blowstats["ProjectileStats"]["active?"] = False
                    Blowstats["ProjectileStats"]["SPD"] += 0.0002
                    if character == TigerStats:
                        Blowstats["ProjectileStats"]["SPD"] += 0.0002
                playerX += playervX
                playerY += playervY
                if character == BrainCrampStats:
                    playervX = 0
                    playervY = 0
                if tick == ticksperframe:
                    if playervX != 0:
                        if playervX < 0:
                            playervX += character["Friction"]
                            round(playervX, 2)
                    if playervX != 0:
                        if playervX > 0:
                            playervX -= character["Friction"]
                            round(playervX, 2)
                        if playervX != 0:
                            if playervX < 0:
                                playervX += character["Friction"]
                                round(playervX, 2)
                    if playervY != 0:
                        if playervY < 0:
                            playervY += character["Friction"]
                            round(playervY, 2)
                    if playervY != 0:
                        if playervY > 0:
                            playervY -= character["Friction"]
                            round(playervY, 2)
                    tick = 0
                    framecounter += 1
                if Blowstats["ProjectileStats"]["x"] < playerX + 60 and Blowstats["ProjectileStats"]["x"] > playerX:
                    if Blowstats["ProjectileStats"]["y"] > playerY and Blowstats["ProjectileStats"]["y"] < playerY + 60:
                        currentHP -= Blowstats["ProjectileStats"]["DMG"]
                        Blowstats["ProjectileStats"]["active?"] = False
                        Blowstats["ProjectileStats"]["x"] = 200000
                        Blowstats["ProjectileStats"]["y"] = 200000
                        #print(currentHP)
                        if character == PeacockStats:
                            character["SPD"] += 0.1 * Blowstats["ProjectileStats"]["DMG"]
                        if character == KangarooStats:
                            character["shotCounter"] = 0
                        if character == BrainCrampStats:
                            if character["shotCounter"] == 0:
                                currentHP += Blowstats["ProjectileStats"]["DMG"]
                if playerX < 250:
                    playerX = 250
                if playerX > window_x - 70:
                    playerX = window_x - 70
                if playerY < 0:
                    playerY = 0
                if playerY > window_y - 70:
                    playerY = window_y - 70
                if currentHP == 0:
                    if character == OrcaStats:
                        character = random.choice(charlist)
                        currentHP = character["HP"]
                        if character == KangarooStats:
                            character["HP"] = 12
                        character["img"] = pygame.transform.scale(character["img"], (60, 60))
                    elif character == HighlandCowStats:
                        GoldenStats["HP"] = character["trumpets"]
                        character = GoldenStats
                        character["img"] = pygame.transform.scale(character["img"], (60, 60))
                        currentHP = character["HP"]
                    else:
                        print("Game Over! You survived", shotsSurvived, "shots")
                        running = False
                        playing = False
                        pygame.quit()
    elif gameChoice == "Blackjack":
        import random
        import itertools as it
        numbers = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
        suits = ['♠','♥','♦','♣']
        cardattributes = [numbers, suits]
        deck = []
        for i in it.product(*cardattributes):
            deck.append(i)
        hand = []
        dealer = []
        global wins
        wins = 0
        global losses
        losses = 0
        global netmoney
        netmoney = 0
        def play(auto, bet):
            deck = []
            for i in it.product(*cardattributes):
                deck.append(i)
            hand = []
            dealer = []
            global wins
            global losses
            global netmoney
            bet = float(bet)
            bust = {'Player': False, 'Dealer': False}
            Aces = 0
            for x in range(2):
                carddrawn = random.choice(deck)
                hand.append(carddrawn)
                deck.remove(carddrawn)
            for x in range(2):
                carddrawn = random.choice(deck)
                dealer.append(carddrawn)
                deck.remove(carddrawn)
            if auto == False:
                print('Your hand:', hand)
                print("Dealer's hand:", dealer[1])
            handvalue = 0
            for card in range(len(hand)):
                if hand[card][0] != 'A':
                    if hand[card][0] == 'K':
                        handvalue += 10
                    elif hand[card][0] == 'Q':
                        handvalue += 10
                    elif hand[card][0] == 'J':
                        handvalue += 10
                    else:
                        handvalue += int(hand[card][0])
                else:
                    handvalue += 11
                    Aces += 1
                    if handvalue >= 22:
                        handvalue -= 10
                        Aces -= 1
            if auto == False:
                if Aces > 0:
                    print('You have a soft', handvalue)
                else:
                    print('You have a hard', handvalue)
            playing = True
            while playing:
                if auto == True:
                    if handvalue >= 17:
                        choice = 'stick'
                    elif handvalue == 10 or handvalue == 11:
                        choice = 'double down'
                    else:
                        choice = 'hit'
                else:
                    choice = input('What do you do?')
                if choice == 'hit':
                    carddrawn = random.choice(deck)
                    hand.append(carddrawn)
                    deck.remove(carddrawn)
                    if auto == False:
                        print('Your hand:', hand)
                    handvalue = 0
                    for card in range(len(hand)):
                        if hand[card][0] != 'A':
                            if hand[card][0] == 'K':
                                handvalue += 10
                            elif hand[card][0] == 'Q':
                                handvalue += 10
                            elif hand[card][0] == 'J':
                                handvalue += 10
                            else:
                                handvalue += int(hand[card][0])
                            if Aces >= 1:
                                if handvalue >= 22:
                                    handvalue-=10
                                    Aces -= 1
                            else:
                                if handvalue >= 22:
                                    bust['Player']=True
                                    if auto == False:
                                        print('You bust on', handvalue)
                                    playing=False
                        else:
                            handvalue += 11
                            Aces += 1
                            if handvalue >= 22:
                                handvalue -= 10
                                Aces -= 1
                    if auto == False:
                        if Aces > 0:
                            print('You have a soft', handvalue)
                        elif bust['Player'] == False:
                            print('You have a hard', handvalue)
                elif choice == 'stick':
                    playing=False
                elif choice == 'double down':
                    carddrawn = random.choice(deck)
                    hand.append(carddrawn)
                    deck.remove(carddrawn)
                    if auto == False:
                        print('Your hand:', hand)
                    handvalue = 0
                    for card in range(len(hand)):
                        if hand[card][0] != 'A':
                            if hand[card][0] == 'K':
                                handvalue += 10
                            elif hand[card][0] == 'Q':
                                handvalue += 10
                            elif hand[card][0] == 'J':
                                handvalue += 10
                            else:
                                handvalue += int(hand[card][0])
                            if Aces >= 1:
                                if handvalue >= 22:
                                    handvalue-=10
                                    Aces -= 1
                            else:
                                if handvalue >= 22:
                                    bust['Player']=True
                                    if auto == False:
                                        print('You bust on', handvalue)
                                    playing=False
                        else:
                            handvalue += 11
                            Aces += 1
                            if handvalue >= 22:
                                handvalue -= 10
                                Aces -= 1
                    if auto == False:
                        if Aces >= 1:
                            print('You have a soft', handvalue)
                        elif bust['Player'] == False:
                            print('You have a hard', handvalue)
                    playing = False
                else:
                    print('Error. Try again.')
            if bust['Player'] == True:
                if choice == "double down":
                    losses += 2
                    netmoney -= 2*bet
                    if auto == False:
                        print('You Lose (but with TWICE THE PAYOUT!!)')
                else:
                    losses += 1
                    netmoney -= bet
                    if auto == False:
                        print('You Lose')
                
            else:
                if auto == False:
                    print(dealer)
                dealerturn=True
                Aces = 0
                while dealerturn:
                    dealervalue = 0
                    for card in range(len(dealer)):
                        if dealer[card][0] != 'A':
                            if dealer[card][0] == 'K':
                                dealervalue += 10
                            elif dealer[card][0] == 'Q':
                                dealervalue += 10
                            elif dealer[card][0] == 'J':
                                dealervalue += 10
                            else:
                                dealervalue += int(dealer[card][0])
                        else:
                            dealervalue += 11
                            Aces += 1
                        if Aces >= 1:
                            if dealervalue >= 22:
                                dealervalue-=10
                                Aces -= 1
                        else:
                            if dealervalue > 21:
                                bust['Dealer']=True
                                if auto == False:
                                    print('The dealer bust on', dealervalue)
                                dealerturn=False
                    if bust['Dealer'] == False:
                        if dealervalue >= 17:
                            dealerturn=False
                            if auto == False:
                                print('Dealer is currently at', dealervalue)
                        else:
                            carddrawn = random.choice(deck)
                            dealer.append(carddrawn)
                            deck.remove(carddrawn)
                            if auto == False:
                                print('Dealer is currently at', dealervalue)
                                print("Dealer's hand:", dealer)
                if bust['Dealer'] == True:
                    if choice == 'double down':
                        if auto == False:
                            print('You Win (but with TWICE THE PAYOUT!!)')
                        wins += 2
                        netmoney += 2*bet
                    else:
                        if auto == False:
                            print('You Win')
                        wins += 1
                        netmoney += bet
                elif dealervalue >= handvalue:
                    if choice == 'double down':
                        if auto == False:
                            print('You Lose (but with TWICE THE PAYOUT!!)')
                        losses += 2
                        netmoney -= 2*bet
                    else:
                        if auto == False:
                            print('You Lose')
                        losses += 1
                        netmoney -= bet
                else:
                    if choice == 'double down':
                        if auto == False:
                            print('You Win (but with TWICE THE PAYOUT!!)')
                        wins += 2
                        netmoney += 2*bet
                    else:
                        if auto == False:
                            print('You Win')
                        wins += 1
                        netmoney += bet
        choosing = True
        while choosing:
            choose = input("Do you want to play or autoplay?")
            betting = True
            while betting:
                bet = input("How much do you want to bet? (float in dollars please)")
                if choose == 'play':
                    play(False, bet)
                    if netmoney < 0:
                        print("So far, you've lost $","%.2f"%(-netmoney))
                    else:
                        print("So far, you've won $","%.2f"%netmoney)
                    again = input("Do you want to play again? (y/n)")
                    if again == 'n':
                        choosing = False
                        betting = False
                elif choose == 'autoplay':
                    reps = input("How many repetitions do you want to do?")
                    reps = int(reps)
                    for x in range(reps):
                        play(True, bet)
                    print('You won',100*wins/(wins+losses),'% of the time')
                    if netmoney < 0:
                        print("You lost $","%.2f"%(-netmoney))
                    else:
                        print("You won $", "%.2f"%netmoney)
                    betting = False
                    choosing = False
                else:
                    print("Seriously?")
                    betting = False

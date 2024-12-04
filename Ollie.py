def main():
    choice = input("Press and enter 1 for a platformer game, press and enter 2 for a guessing game, press and enter 3 for code about dice probabilities, press and enter 4 for code about birthdays")
    if choice == "1":
        import pygame
        import sys
        import random

        print("Keys are left and right arrow keys and the space bar to jump.")

        # Initialize Pygame
        pygame.init()
        pygame.font.init()

        # Screen settings
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simple Platformer")
        outerlimit = -600
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render('You Died!', False, (255, 0, 0))
        platspeed = 1
        groundplatx = 800
        clockspeed = 60
        realclock = 0
        roundclock = 0
        clockstop = False
        # Colors
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        
        # Game settings
        GRAVITY = 0.5
        PLAYER_JUMP_STRENGTH = 13
        PLAYER_SPEED = 5

        # Player settings
        player_width, player_height = 40, 50
        player_x, player_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT - player_height
        player_velocity_x, player_velocity_y = 0, 0
        on_ground = False
        pushedback = False
        playerdead = False

        jumper = 0

        player_x = 100
        player_y = 100



        # Main game loop
        clock = pygame.time.Clock()
        counter = 1
        while True:     
            # Event handling
            if counter == 1:
                #block 1
                platxrandom1 = random.randint(1, 600)
                platyrandom1 = random.randint(350, 600)
                platwrandom1 = random.randint(1, 600)
                plathrandom1 = random.randint(1, 600)

                platxrandom2 = random.randint(1, 600)
                platyrandom2 = random.randint(350, 600)
                platwrandom2 = random.randint(1, 600)
                plathrandom2 = random.randint(1, 600)

                platxrandom3 = random.randint(1, 600)
                platyrandom3 = random.randint(350, 600)
                platwrandom3 = random.randint(1, 600)
                plathrandom3 = random.randint(1, 600)

                platxrandom4 = random.randint(1, 600)
                platyrandom4 = random.randint(350, 600)
                platwrandom4 = random.randint(1, 600)
                plathrandom4 = random.randint(1, 600)
        
                #air platforms
        
                platxrandom5 = random.randint(1, 600)
                platyrandom5 = random.randint(100, 350)
                platwrandom5 = random.randint(1, 600)
                plathrandom5 = random.randint(1, 130)

                platxrandom6 = random.randint(1, 600)
                platyrandom6 = random.randint(100, 350)
                platwrandom6 = random.randint(1, 600)
                plathrandom6 = random.randint(1, 130)

                platxrandom7 = random.randint(1, 600)
                platyrandom7 = random.randint(100, 350)
                platwrandom7 = random.randint(1, 600)
                plathrandom7 = random.randint(1, 130)

                #block 2

                platxrandom8 = (random.randint(1, 600) + 1200)
                platyrandom8 = random.randint(350, 600)
                platwrandom8 = random.randint(1, 600)
                plathrandom8 = random.randint(1, 600)

                platxrandom9 = (random.randint(1, 600) + 1200)
                platyrandom9 = random.randint(350, 600)
                platwrandom9 = random.randint(1, 600)
                plathrandom9 = random.randint(1, 600)

                platxrandom10 = (random.randint(1, 600) + 1200)
                platyrandom10 = random.randint(350, 600)
                platwrandom10 = random.randint(1, 600)
                plathrandom10 = random.randint(1, 600)

                platxrandom11 = (random.randint(1, 600) + 1200)
                platyrandom11 = random.randint(350, 600)
                platwrandom11 = random.randint(1, 600)
                plathrandom11 = random.randint(1, 600)

                #air platforms

                platxrandom12 = (random.randint(1, 600) + 1200)
                platyrandom12 = random.randint(100, 350)
                platwrandom12 = random.randint(1, 600)
                plathrandom12 = random.randint(1, 130)

                platxrandom13 = (random.randint(1, 600) + 1200)
                platyrandom13 = random.randint(100, 350)
                platwrandom13 = random.randint(1, 600)
                plathrandom13 = random.randint(1, 130)
                
                platxrandom14 = (random.randint(1, 600) + 1200)
                platyrandom14 = random.randint(100, 350)
                platwrandom14 = random.randint(1, 600)
                plathrandom14 = random.randint(1, 130)
                
            counter += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Key states
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player_velocity_x = -PLAYER_SPEED
            elif keys[pygame.K_RIGHT] and pushedback == False:
                player_velocity_x = PLAYER_SPEED
            else:
                player_velocity_x = 0

            # Jumping
            if keys[pygame.K_SPACE] and on_ground:
                player_velocity_y = -PLAYER_JUMP_STRENGTH
                on_ground = False

                    
            # Platform settings
            platforms = [
                #block 1
                pygame.Rect(platxrandom1, platyrandom1, platwrandom1, plathrandom1), #x, y, width, height
                pygame.Rect(platxrandom2, platyrandom2, platwrandom2, plathrandom2),
                pygame.Rect(platxrandom3, platyrandom3, platwrandom3, plathrandom3),
                pygame.Rect(platxrandom4, platyrandom4, platwrandom4, plathrandom4),
                pygame.Rect(platxrandom5, platyrandom5, platwrandom5, plathrandom5),
                pygame.Rect(platxrandom6, platyrandom6, platwrandom6, plathrandom6),
                pygame.Rect(platxrandom7, platyrandom7, platwrandom7, plathrandom7),
                #block 2
                pygame.Rect(platxrandom8, platyrandom8, platwrandom8, plathrandom8), #x, y, width, height
                pygame.Rect(platxrandom9, platyrandom9, platwrandom9, plathrandom9),
                pygame.Rect(platxrandom10, platyrandom10, platwrandom10, plathrandom10),
                pygame.Rect(platxrandom11, platyrandom11, platwrandom11, plathrandom11),
                pygame.Rect(platxrandom12, platyrandom12, platwrandom12, plathrandom12),
                pygame.Rect(platxrandom13, platyrandom13, platwrandom13, plathrandom13),
                pygame.Rect(platxrandom14, platyrandom14, platwrandom14, plathrandom14),
                pygame.Rect(groundplatx, 400, 300, 20)
            ]
            evilplatforms = [pygame.Rect(0, 0, 20, SCREEN_HEIGHT)]  #DEATHWALL!!!!

            platxrandom1 -= platspeed
            platxrandom2 -= platspeed
            platxrandom3 -= platspeed
            platxrandom4 -= platspeed
            platxrandom5 -= platspeed
            platxrandom6 -= platspeed
            platxrandom7 -= platspeed
            platxrandom8 -= platspeed
            platxrandom9 -= platspeed
            platxrandom10 -= platspeed
            platxrandom11 -= platspeed
            platxrandom12 -= platspeed
            platxrandom13 -= platspeed
            platxrandom14 -= platspeed
            groundplatx -= platspeed
            
            if platxrandom1 < outerlimit and platxrandom2 < outerlimit and platxrandom3 < outerlimit and platxrandom4 < outerlimit and platxrandom5 < outerlimit and platxrandom6 < outerlimit and platxrandom7 < outerlimit:
                platxrandom1 = (random.randint(1, 600) + 900)
                platyrandom1 = random.randint(350, 600)
                platwrandom1 = random.randint(1, 600)
                plathrandom1 = random.randint(1, 600)

                platxrandom2 = (random.randint(1, 600) + 900)
                platyrandom2 = random.randint(350, 600)
                platwrandom2 = random.randint(1, 600)
                plathrandom2 = random.randint(1, 600)

                platxrandom3 = (random.randint(1, 600) + 900)
                platyrandom3 = random.randint(350, 600)
                platwrandom3 = random.randint(1, 600)
                plathrandom3 = random.randint(1, 600)

                platxrandom4 = (random.randint(1, 600) + 900)
                platyrandom4 = random.randint(350, 600)
                platwrandom4 = random.randint(1, 600)
                plathrandom4 = random.randint(1, 600)
                
                platxrandom5 = (random.randint(1, 600) + 900)
                platyrandom5 = random.randint(100, 350)
                platwrandom5 = random.randint(1, 600)
                plathrandom5 = random.randint(1, 130)
                
                platxrandom6 = (random.randint(1, 600) + 900)
                platyrandom6 = random.randint(100, 350)
                platwrandom6 = random.randint(1, 600)
                plathrandom6 = random.randint(1, 130)
                
                platxrandom7 = (random.randint(1, 600) + 900)
                platyrandom7 = random.randint(100, 350)
                platwrandom7 = random.randint(1, 600)
                plathrandom7 = random.randint(1, 130)  

            if platxrandom8 < outerlimit and platxrandom9 < outerlimit and platxrandom10 < outerlimit and platxrandom11 < outerlimit and platxrandom12 < outerlimit and platxrandom13 < outerlimit and platxrandom14 < outerlimit:   
                platxrandom8 = (random.randint(1, 600) + 900)
                platyrandom8 = random.randint(350, 600)
                platwrandom8 = random.randint(1, 600)
                plathrandom8 = random.randint(1, 600)

                platxrandom9 = (random.randint(1, 600) + 900)
                platyrandom9 = random.randint(350, 600)
                platwrandom9 = random.randint(1, 600)
                plathrandom9 = random.randint(1, 600)

                platxrandom10 = (random.randint(1, 600) + 900)
                platyrandom10 = random.randint(350, 600)
                platwrandom10 = random.randint(1, 600)
                plathrandom10 = random.randint(1, 600)

                platxrandom11 = (random.randint(1, 600) + 900)
                platyrandom11 = random.randint(350, 600)
                platwrandom11 = random.randint(1, 600)
                plathrandom11 = random.randint(1, 600)
                
                #air platforms
                
                platxrandom12 = (random.randint(1, 600) + 900)
                platyrandom12 = random.randint(100, 350)
                platwrandom12 = random.randint(1, 600)
                plathrandom12 = random.randint(1, 130)
                
                platxrandom13 = (random.randint(1, 600) + 900)
                platyrandom13 = random.randint(100, 350)
                platwrandom13 = random.randint(1, 600)
                plathrandom13 = random.randint(1, 130)
                
                platxrandom14 = (random.randint(1, 600) + 900)
                platyrandom14 = random.randint(100, 350)
                platwrandom14 = random.randint(1, 600)
                plathrandom14 = random.randint(1, 130)
                


            # Gravity
            player_velocity_y += GRAVITY

            # Update player position
            player_x += player_velocity_x
            player_y += player_velocity_y

            # Boundary checks
            if player_x < 0:
                player_x = 0
            elif player_y > 600:
                playerdead = True
            elif player_x + player_width > SCREEN_WIDTH:
                player_x = SCREEN_WIDTH - player_width

            # Platform collision detection
            player_rect = pygame.Rect(player_x, player_y, player_width, player_height)
            on_ground = False
            pushedback = False
            for platform in platforms:
                if player_rect.colliderect(platform):
                    # Determine the side of collision
                    if player_velocity_y > 0 and player_rect.bottom > platform.top and player_rect.bottom - player_velocity_y <= platform.top:
                        # Land on top of platform
                        player_y = platform.top - player_height
                        player_velocity_y = 0
                        on_ground = True
                    elif player_velocity_y < 0 and player_rect.top < platform.bottom and player_rect.top - player_velocity_y >= platform.bottom:
                        # Hit the bottom of platform
                        player_y = platform.bottom
                        player_velocity_y = 0
                    elif player_velocity_x > 0 and player_rect.right > platform.left and player_rect.right - player_velocity_x <= platform.left:
                        # Hit the left side of platform
                        player_x = ((platform.left) - player_width) - (player_velocity_x -4.95)
                        on_ground = True
                    elif player_velocity_x == 0 and player_rect.right > platform.left and player_rect.right - (player_velocity_x + 1) <= platform.left:
                        # Hit the left side of platform
                        player_x -= 1
                        pushedback = True
                        on_ground = True
                    elif player_velocity_x < 0 and player_rect.left < platform.right and player_rect.left - player_velocity_x >= platform.right:
                        # Hit the right side of platform
                        player_x = platform.right
                        player_velocity_x = 0
                        on_ground = True
            for platform in evilplatforms:
                if player_velocity_x <= 0 and player_rect.left < platform.right and player_rect.left - player_velocity_x >= platform.right:
                        # Hit the right side of platform
                        playerdead = True
                elif player_velocity_x >= 0 and player_rect.left < platform.right and player_rect.left - ( player_velocity_x - 1) >= platform.right:
                        # Hit the right side of platform
                        playerdead = True


            # Redefine player_rect to account for updated position
            player_rect = pygame.Rect(player_x, player_y, player_width, player_height)

            # Drawing
            screen.fill(WHITE)  # Clear screen with white background
                
            # Draw platforms
            for platform in platforms:
                pygame.draw.rect(screen, GREEN, platform)
            for platform in evilplatforms:
                pygame.draw.rect(screen, RED, platform)
            

            # Draw player
            pygame.draw.rect(screen, BLUE, player_rect)
                
            if playerdead == True:
                screen.fill((0,0,0))
                pygame.display.flip()
                screen.blit(text_surface, (325, 250))

            # Update display
            pygame.display.flip()
            clock.tick(clockspeed)# Limit to 60 frames per second
    if choice == "2":
        import random
        print("I will pick a number between 1 and: (Type the number)")
        choice = float(input())

        num = random.randint(1, choice)
        guess = False
        guessnum = 0
        while guess == False:
            print("Guess a number between 1 and", choice)
            trier = int(input())
            if trier == num:
                guess = True
                guessnum += 1
            elif trier >= int(num):
                print("Too high!")
                guessnum += 1
            elif trier <= num:
                print("Too low!")
                guessnum += 1
        if guess == True:
            print("Great job! You took",  guessnum, "tries.")
    if choice == "3":
        import random
        import time

        print("You are rolling two dice to find the probability of the summed result.")
        time.sleep(2)
        print("Rolling...")
        time.sleep(2)

        die1 = random.randint(1,6)
        die2 = random.randint(1,6)
        diesum = die1 + die2

        dietotal = 36
        dieprob = 0

        if diesum == 2:
            dieprob += 1
        if diesum == 3:
            dieprob += 2
        if diesum == 4:
            dieprob += 3
        if diesum == 5:
            dieprob += 4
        if diesum == 6:
            dieprob += 5
        if diesum == 7:
            dieprob += 6
        if diesum == 8:
            dieprob += 5
        if diesum == 9:
            dieprob += 4
        if diesum == 10:
            dieprob += 3
        if diesum == 11:
            dieprob += 2
        if diesum == 12:
            dieprob += 1  

        print("Your result is", die1, "and", die2,",", "or", diesum)

        print("The probability of this result is", dieprob, "/", dietotal)
    if choice == "4":
        import time
        from datetime import datetime
        from datetime import date
        current_time = datetime.now()
        import os

        def clear_console():
            """Clears the console."""
            command = "cls" if os.name == "nt" else "clear"
            os.system(command)
            
        def get_difference(date1, date2):
            delta = date2 - date1
            return delta.days

        YEAR, MONTH, DAY = current_time.year, current_time.month, current_time.day

        BYEAR = int(input("What is your birth year?"))

        BMONTH = int(input("What is your birth month, in a number?"))

        BDAY = int(input("What is your birth day?"))


        d1 = date(BYEAR, BMONTH, BDAY)
        d2 = date(YEAR, MONTH, DAY)
        days = get_difference(d1, d2)

        print("Would you like to see your age in seconds, minutes, hours, days, or years?")
        c1 = input()
        if c1 == "seconds":
            age = days * 86400
            agefull = age + (current_time.second) + (current_time.minute * 60) + (current_time.hour * 3600)
            while True:
                agefull += 1
                clear_console()
                print("You have been alive for", str(agefull), "seconds")
                time.sleep(1)
        elif c1 == "minutes":
            age = days * 1440
            agefull = age + (current_time.second / 60) + (current_time.minute) + (current_time.hour * 60)
            while True:
                agefull += 0.01666666666
                clear_console()
                print("You have been alive for", str(agefull), "minutes")
                time.sleep(1)
        elif c1 == "hours":
            age = days * 24
            agefull = age + (current_time.second / 3600) + (current_time.minute /60) + (current_time.hour)
            while True:
                agefull += 0.00027777777
                clear_console()
                print("You have been alive for", str(agefull), "hours")
                time.sleep(1)        
        elif c1 == "days":
            age = days
            agefull = age + (current_time.second / 86400) + (current_time.minute / 1440) + (current_time.hour / 24)
            while True:
                agefull += 0.00001157407
                clear_console()
                print("You have been alive for", str(agefull), "days")
                time.sleep(1)
        elif c1 == "years":
            age = days / 365
            agefull = age + (current_time.second / 31540000) + (current_time.minute / 525600) + (current_time.hour / 8760)
            while True:
                agefull += 0.00001157407
                clear_console()
                print("You have been alive for", str(agefull), "years")
                time.sleep(1)
main()


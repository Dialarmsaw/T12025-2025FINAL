# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:11:37 2024

@author: evanj
"""

def main():
  import os
  from random import randrange
  from time import time


  def hangman():
      words_list = ["galaxy", "xylophone", "zebra", "telephone", "schizophrenia", "kumquat", "refrigerator", "burlap", "highlighting", "necklace", "extraordinary", "chromozome", "fish", "computer", "thumbdrive", "bling", "mascara", "bubble", "infection", "injection", "organic", "kale", "graph", "lick", "eyelash", "aquarium", "haircut", "cuttlefish", "earwax", "psychopathic", "crayon", "vitamin", "diaphragm", "giggle", "crumb", "awkward", "worship", "absurd", "jazz", "pharaoh", "weird", "reindeer", "neighbor", "seaweed", "spider", "fly", "mathematician", "catfish", "quiz", "invite", "chaos", "chimeney", "gorgonzola", "cheese", "ball", "syntax", "passion", "papaya", "cactus", "elegant", "bizarre", "unbeatable", "skier", "junk", "marmelade", "puff", "lychee", "lichen", "chicken", "hangman", "lizard", "alcohol", "chemistry", "physics", "virtue", "patience", "exception", "accept", "except", "exempt" "reflux", "reflex", "murder", "fuzzy", "nebula", "quick", "yak", "yellow", "defective", "deficient", "dysphagia", "dreadful", "disaster", "goose", "guru", "bed", "cinnamon", "banana", "barbeque",  "chocolate", "candy", "pinky", "thumb", "bumblebee", "toothpick", "humidifier", "puppy", "squirmish", "worm", "fantastic", "toilet", "ukulele", "hockey", "biology", "mystery", "zumba", "dizziness"]

      words_mc = ["enderman","endermite","creeper","guardian","elder guardian","drowned","husk","iron golem","mooshroom","ravager","stray","vex","vindicator","evoker","pillager","ghast","warden","blaze","wither","wither skeleton","piglin","piglin brute","hoglin","zoglin","strider","silverfish","magma cube","slime","ender dragon","shulker","cod","allay","zombie","skeleton","camel","axolotl","polar bear","wolf","dolphin","wandering trader","frog","ocelot","sniffer","turtle","overworld","nether","lush cave","deep dark biome","pillager outpost","warped forest","crimson forest","bastion remnant","nether fortress","deep dark","stronghold","enchanting table","redstone","prismarine","dark prismarine","blackstone","gilded blackstone","glowstone","soul sand","soul soil","shroomlight","netherite","ancient debris","netherrack","nether wart","ender pearl","eye of ender","endstone","end crystal","elytra","end city","purpur","chorus plant","chorus fruit","ol betsy","return to sender","we need to go deeper","ice bucket challenge","who is cutting onions","not today thank you", "indev","infdev","nether update","caves and cliffs","pretty scary update","aquatic update","village and pillage","end update","buzzy bees","redstone update"]

      words_brainrot = ["skibidi","skibidi toilet","sigma","erm what the sigma","gyatt","ohio","only in ohio","rizz","rizzler","alpha","beta","fanum tax","sticking out your gyatt for the rizzler youre so skibidi youre so fanum tax","aura","negative aura","caseoh","the fitnessgram pacer test","amogus","sus","sussy impostor","mindful","demure","hitting the griddy","goofy ahh","one two buckle my shoe","grimace shake","l bozo","get ratioed","english or spanish","looksmaxxing","andrew tate","gigachad","bing chilling","balkan rage","stillwater","troll face","brat summer","the bite of eighty seven","ishowspeed","kai cenat","adin ross","aaaaaaaaaaah","disney minus","rickroll","double it and give it to the next person","jebaited","freddy fazbear","john pork","treat me like white tees","chungus","smurf cat","poggers","dj khaled","quandale dingle","morbin time","mewing","ugandan knuckles","backrooms","metal pipe falling","samsung notification","me after eating taco bell","presidents playing minecraft","family guy funny moments compilation with subway surfers gameplay at the bottom"]

      theme = input("Select a theme:\n\n1) Hard\n2) Minecraft\n3) Brainrot\n4) Custom word\n\n>>> ")
      valid = False
      while not valid:
        if theme in ["1","2","3","4"]:
          valid = True
        else:
          theme = input("Invalid input\n>>> ")
      theme = int(theme)
      theme_options = ["Hard mode", "Minecraft mode", "Brainrot mode", "Custom word"]
      themes = [words_list,words_mc,words_brainrot]
      if theme != 4:
        dictionary = themes[theme-1]
      print()

      alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
      if theme == 4:
        acceptable_word = [False]
        word = ""
        first_time = True
        while False in acceptable_word or word.count(" ")==len(word): # if there are illegal characters or it's all spaces, either attempt to take out illegal characters or reprompt
          acceptable_word = []
          if first_time:
            word = input("Enter a word or phrase for a friend to guess. >>> ")
          else:
            word = input("Invalid input. Must be only lowercase letters or spaces. >>> ")
          for i in range(len(word)):
            acceptable_word.append(bool(word[i] in alphabet or word[i] == " ")) 
          first_time = False
      else:
        word = dictionary[randrange(len(dictionary))] # choose word
      
      word_new = ""
      for i in range(len(word)): # replace multiple spaces in a row with just one space.
          if not ((word[i] == " " and word[i-1] == " ") or (word[i] == " " and i == 0)): # when should we do nothing? when 1) both this character is a space and the last one was also a space, or 2) if this character is a space and it is the first character. when it is neither, copy over the character. it doesn't matter if the last character of the entire word is a space becuase that won't be seen by the user.
              word_new += word[i]
      word = word_new

      again = True      # variable that figures out if user has lost or won. Each iteration of the loop is an entire game
      guessList = []    # list of letters user has previously guessed
      hidden_word = []  # list of "_" that hides the word
      hint_given = False
      for i in range(len(word)): # build list of "_"
          hidden_word.append("  " if word[i] == " " else "_")

      ###

      while again:     
        head = 0 # setting the number of body parts the victim has, 0 at beginning of game
        body = 0
        arms = 0
        legs = 0
        eyes = 0
        nose = 0
        mouth = 0

        lost = False
        os.system('cls' if os.name == 'nt' else 'clear') # clear theme dialogues and clear between games (if user wants to try again with the same word)

        #Guessing
        
        while ("_" in hidden_word) and not lost: # while user has not won or lost yet
          print(theme_options[theme-1],"\n") # print header, hidden word, and body parts
          print(*hidden_word)
          print("\nHead:", str(head) + "/1\nBody:", str(body) + "/1\nArms:", str(arms) + "/2\nLegs:", str(legs) + "/2\nBoth eyes:", str(eyes) + "/1\nNose:", str(nose) + "/1\nMouth:", str(mouth) + "/1\n")
          
          lost = bool(mouth==1)   
          acceptable_guess = False
          if not lost:
            if len(guessList) != 0: # printing out letters user has already guessed
              print("You've previously guessed: ", end="")
              print(*guessList, sep=", ")
              
            while not acceptable_guess:
              if not hint_given:
                guess = input('Type in a letter that you want to guess, or type "/" for a hint. >>> ')
                if guess == "/": # give hint
                  hint_letter = " " # force into while loop
                  while hint_letter in hidden_word or hint_letter not in alphabet: # if the letter has already been guessed, choose a different hint_letter
                    hint_letter = word[randrange(len(word))] # choose random letter to reveal
                  guess = hint_letter
                  hint_given = True
              else:
                guess = input("Type in a letter that you want to guess. >>> ")
              if guess in alphabet: # check if the guess counts, and if not, re-prompt user for a letter without adding a body part
                if guess in guessList:
                    print("You've already guessed", guess)
                else:
                  guessList.append(guess)
                  acceptable_guess = True
              elif guess != "/":
                print("Invalid guess")
              elif guess == "/" and hint_given:
                print("You already got a hint.")
            os.system('cls' if os.name == 'nt' else 'clear') # clear between each valid guess
            
            if guess in word:
              for i in range(len(word)):          # if the guess is in the word, then put it in, repeat if guess appears more than once
                if word[i] == guess:
                  hidden_word[i] = guess
            else:
              if head == 0:                       # if the guess isn't in the word, add a body part
                head+=1                           
              elif body == 0:
                body+=1
              elif arms == 0 or arms == 1:
                arms+=1
              elif legs == 0 or legs == 1:
                legs+=1
              elif eyes == 0:
                eyes+=1
              elif nose == 0:
                nose+=1
              elif mouth == 0:
                mouth+=1
            if "_" not in hidden_word:            # if all the letters have been guessed, user won!!!
              print(theme_options[theme-1],"\n")
              print(*hidden_word, "\n\nCorrect!\nYou got it in", len(guessList), "guesses!")
              again = False                       # don't repeat again, bc user won
          
          else: # if player has lost
            print("Oops...\nYou lost!") # if victim is hung and user has not yet finished the word, user lost...
            again_input = input("Would you like to try again with the same word? (y/n) >>> ") # give user another chance
            while again_input not in ["y","n"]:
              again_input = input("Invalid input (y/n) >>> ")
            if again_input == "y":                                                            
              again = True # if user says yes, reset everything and start over, but with the same word
            else:
              again = False # if no, don't repeat again
              print("\nIt was '", word, "'!", sep="") # reveal word
              
      END = input("\n1) Exit script\n2) Run again\n\n>>> ")
      valid = False
      while not valid:
          if END not in ["1","2"]:
              END = input("Invalid input.\n>>> ")
          else:
              valid = True
      if END == "2":
          os.system('cls' if os.name == 'nt' else 'clear')
          hangman()
      
      os.system('cls' if os.name == 'nt' else 'clear')



  def montecarlo():
      print("You randomly take a certain number of cards out of a shuffled deck of 52 cards\nand throw them in the trash. Then, what's the probability that all four of one\nnumber is still in the deck?\n")
      
      result = []
        
      valid = False
      first_time = True
      while not valid: # safe to turn into an integer?
          try:
              num_deal = int(input("How many cards to throw in the trash?\n>>> " if first_time else "Invalid input. Must be a number.\n>>> "))
              Valid = False
              while not Valid:
                  if num_deal < 1 or num_deal > 52:
                      num_deal = int(input("Invalid input. Must be between 1 and 52.\n>>> "))
                  else:
                      Valid = True
              valid = True
          except Exception:
              first_time = False
              
      valid = False
      first_time = True
      while not valid: # safe to turn into an integer?
          try:
              iters = int(input("Now, we need to run the simulation over and over again. The more times we run\nit, the probability becomes more accurate, but it takes longer to compute.\nHow many times to run it?\n>>> " if first_time else "Invalid input. Must be a number.\n>>> "))
              Valid = False
              while not Valid:
                  if iters < 1 or iters > 1000000:
                      iters = int(input("Invalid input. Must be between 1 and 1 million.\n>>> "))
                  else:
                      Valid = True
              valid = True
          except Exception:
              first_time = False
              
      print("Please wait...")
      
      t0 = time()

      for _ in range(iters):
          deck = [4] * 13
          
          for _ in range(num_deal): # deal 8 cards, then deal half the remaining deck 8+((52-8)/2)=30
              num = randrange(len(deck)) # num = number of the card, deck[num] = how many of that card there are
              while deck[num] == 0:
                  num = randrange(len(deck)) # choose different num if all 4 cards of that num have already been dealt
              deck[num] -= 1 # choose random card to remove (deal) from deck
          
          i = 0
          try:
              while deck[i] != 4: # while all 4 numbers haven't been found yet
                  i += 1
              result.append(1) # all four of one number exists
          except Exception: # if it has checked kings and there still isn't all four of one number
              result.append(0) # all four of one number doesn't exist
      
      t1 = time() - t0
      print("\nProbability (out of 1):", sum(result)/len(result))
      print("That took", t1, "seconds.")
      
      END = input("\n1) Exit script\n2) Run again\n\n>>> ")
      valid = False
      while not valid:
          if END not in ["1","2"]:
              END = input("Invalid input.\n>>> ")
          else:
              valid = True
      if END == "2":
          os.system('cls' if os.name == 'nt' else 'clear')
          montecarlo()
      os.system('cls' if os.name == 'nt' else 'clear')
      


  
  select = input("Select a script to run:\n\n1) Hangman\n2) Dealing cards\n\n>>> ")
  if select == "1":
    hangman()
  elif select == "2":
    montecarlo()
  else:
    select = input("Invalid input.\n>>> ")
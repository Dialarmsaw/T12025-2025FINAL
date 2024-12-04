def main():
    if input("Press and enter 1 for madlibs, press and enter 2 for code about the probability of not getting a heart when you draw from a deck") == "1":
        print("Welcome to MadLibs! In this game, I will ask you for a series of words to create a story. Let’s begin!")

        Noun1 = input("Type a noun: 	")
        Noun2 = input("Type a noun: 	")
        Verb1 = input("Type a verb: 	")
        Verb2 = input("Type a verb: 	")
        Food1 = input("Type a food: 	")
        noun3 = input("Type a noun:     ")
        adjetive1 = input("type an adjetive:     ")
        emotion1 = input("type an emotion:     ")
        timeperiod = input("type a timeperiod:   ")
        print ("Once upon a time there was a", 	Noun1, "who lived with", Noun2, ". one day", Noun1, "decided to", Verb1, "! while", Noun1, "was doing this", Noun2, "started to", Verb2, "while eating", Food1, ".", "Suddently, a", adjetive1, noun3, "came twards me.", "I was", emotion1, ".", "That night, I thought about the", adjetive1, noun3, "as I went to bed." "That", adjetive1, noun3, "would haunt me for", timeperiod, ".")
    else:
        import random

        def deck():
            suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
            rand = random.choice(suits)
            return rand
        #suit1 = input("suit1:")
        #suit2= input("suit2:")
        #suit3 = input("suit3:")

        count = 0

        for n in range (100000):
             d1 = deck()
             d2 = deck()
             d3 = deck()
             if d1 == "club" and d2 == "diamond" and d3 == "spade":
                 count+=1
                # print("you got it!")
                # print (count)

        print (count/100000*100)
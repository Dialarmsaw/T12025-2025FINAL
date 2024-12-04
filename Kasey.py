def main():
    print("Hi! welcome to Kasey's section!")

    s = int(input("to select Mad Libs press 1, to select A montecarlo code press 2"))


    if s == 1:
        print("Welcome to MadLibs! We're going to play a game where you give the words and I tell you a story!")

        NOUN = input("Give a NOUN related to TIME, for example yesterday")

        ADJ = input ("Give an ADJECTIVE related to WEATHER")

        NAME = input ("Give a NAME")

        VERB = input ("Give a VERB")

        COL = input ("Give a COLOR")

        NUM = input ("Give a NUMBER")

        ADV = input ("Give a ADVERB")

        print (NOUN, "it was very", ADJ, "and", NAME, "decided to", VERB, "outside. The trees were", COL, "and there were", NUM, "of them, to", NAME, "they seemed to move", ADV)



    import random

    if s == 2:
        print ("If you roll two 6-sided dice, what is the probability that you roll a 2 and a 4?")

        success = 0
        amount = 10
        for _ in range(amount):
            di = int(random.randrange(1, 7))
            print(di)
            dice = int(random.randrange(1, 7))
            print(dice)
            if dice == 2 and di == 4:
                success = success+1
                print ("success!")
            elif di == 2 and dice == 4:
                success = success+1
                print ("success!")
            else:
                print("fail")


        print (f"the probability of rolling a 2 and 4 is {success/amount}!")


